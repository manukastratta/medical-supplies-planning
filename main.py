import numpy as np
from scipy.stats import norm
import random
from itertools import chain, combinations
import csv
import timeit

# for plotting
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

class Final_Project:
    def __init__(self):
        self.NUM_HOSPITALS = 10

        # For Q-learning
        self.NUM_EPOCHS = 100
        self.N_SAMPLES = 5
        self.lr = .1
        self.DISCOUNT_FACTOR = .95 #.95 for small and large, 1 for medium

        # For the reward function
        self.DISTANCE_WEIGHT = 1
        self.URGENCY_WEIGHT = 1
        self.URGENCY_SCALE = .15
    
        # Stores a mapping from every state to an index (integer)
        self.state_to_index = dict()

        # Stores the coordinates for a given hospital
        # Note: hospitals start at 1, 0 refers to starting location [states list]
        self.hospital_to_coord = dict() 

        # Stores our learned distributions for blood and vaccines
        self.hospital_to_blood_dist = dict()
        self.hospital_to_vaccine_dist = dict()

        # Generated dataset file from generate_dataset()
        self.in_file = "random_with_urgency_preliminary_dataset.csv"

        # Generated policy file from q_learning()
        self.out_file = "random_with_urgency_100ep.policy" # AKA the policy file

        # Final route
        self.final_route = None
        
        # Set random seeds for math and numpy functions for deterministic results
        random.seed(50)
        np.random.seed(50)

    ###################################################################################
    ############################### REWARD FUNCTION####################################
    ###################################################################################
    def reward(self, curr_state, next_state):
        curr_grid_location = np.array(self.hospital_to_coord[curr_state[0]])
        next_grid_location = np.array(self.hospital_to_coord[next_state[0]])
        distance_comp = int(25 / np.linalg.norm (curr_grid_location - next_grid_location))

        num_visited_nodes = len(curr_state[1]) + 1
        blood_pred = self.hospital_to_blood_dist[next_state[0]][0]
        vaccine_pred = self.hospital_to_vaccine_dist[next_state[0]][0]
        urgency_comp = self.URGENCY_SCALE/num_visited_nodes * (blood_pred + vaccine_pred)

        final_reward = self.DISTANCE_WEIGHT * distance_comp + self.URGENCY_WEIGHT * urgency_comp
        return final_reward

    # OLD REWARD FUNCTION OF ONLY EUCLIDEAN DISTANCE
    # def reward(self, curr_state, next_state):
    #     curr_grid_location = np.array(self.hospital_to_coord[curr_state[0]])
    #     next_grid_location = np.array(self.hospital_to_coord[next_state[0]])
    #     return int(25 / np.linalg.norm (curr_grid_location - next_grid_location))


    ###################################################################################
    ############################### GENERATE DATASET ##################################
    ###################################################################################
    def generate_dataset(self):
        def powerset(s):
            return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

        # # Generate true distributions of weekly hospital orders

        blood_distrs = list()
        vaccine_distrs = list()

        # denotes the x-axis range
        blood_data = np.arange(0, 600, 0.01)
        vaccine_data = np.arange(0, 900, 0.01)

        fig_blood, ax_blood = plt.subplots()
        fig_vaccine, ax_vaccine = plt.subplots()

        for i in range(1, self.NUM_HOSPITALS+1):
            mu_blood, sigma_blood = i*50, random.uniform(3*i, 5*i)
            mu_vaccine, sigma_vaccine = i*100, random.uniform(6*i, 10*i)

            print(f'Blood -> Hospital {i}: {mu_blood}, {sigma_blood}')
            print(f'Vaccine -> Hospital {i}: {mu_vaccine}, {sigma_vaccine}')
            blood_samples = np.random.normal(mu_blood, sigma_blood, self.N_SAMPLES)
            vaccine_samples = np.random.normal(mu_vaccine, sigma_vaccine, self.N_SAMPLES)

            blood_distrs.append(np.rint(blood_samples)) # round to nearest integer
            vaccine_distrs.append(np.rint(vaccine_samples))

            ax_blood.plot(blood_data, norm.pdf(blood_data, scale=sigma_blood, loc=mu_blood), label=f"Hospital #{i}")
            ax_vaccine.plot(vaccine_data, norm.pdf(vaccine_data, scale=sigma_vaccine, loc=mu_vaccine), label=f"Hospital #{i}")

        ax_blood.set_title('True Distribution of Blood Supply Orders')
        ax_blood.legend(loc='best', frameon=True)

        ax_vaccine.set_title('True Distribution of Vaccine Supply Orders')
        ax_vaccine.legend(loc='best', frameon=True)

        # Save hospital past order data, 1 file per hospital
        for i in range(0, self.NUM_HOSPITALS):
            np.savetxt(f'blood_data/hospital{i+1}.txt', blood_distrs[i])
            np.savetxt(f'vaccine_data/hospital{i+1}.txt', vaccine_distrs[i])

        # For every hospital, learn a distribution
        for i in range(0, self.NUM_HOSPITALS):
            blood_mean, blood_std = norm.fit(blood_distrs[i][:])
            self.hospital_to_blood_dist[i+1] = (blood_mean, blood_std)

            vaccine_mean, vaccine_std = norm.fit(vaccine_distrs[i][:])
            self.hospital_to_vaccine_dist[i+1] = (vaccine_mean, vaccine_std)

        print(f'hospital_to_blood_dist: {self.hospital_to_blood_dist}')
        print(f'hospital_to_vaccine_dist: {self.hospital_to_vaccine_dist}')

        # Generate random hospital locations:
        MAX_DIM = 10
        possible_points = []
        for x in range(0, MAX_DIM + 1):
            for y in range(0, MAX_DIM + 1):
                curr_tuple = (x,y)
                possible_points.append(curr_tuple)
        hospital_coords = random.sample(possible_points, self.NUM_HOSPITALS)

        self.hospital_to_coord[0] = (0,0)
        for curr_index in range(1, self.NUM_HOSPITALS + 1): 
            self.hospital_to_coord[curr_index] = hospital_coords[curr_index - 1]
        print(self.hospital_to_coord)

        # state = (curr_state, set of states visited already)
        # Generate powerset / build up state space
        hospital_list = list(range(1, self.NUM_HOSPITALS + 1))
        hospital_set = set(range(1, self.NUM_HOSPITALS + 1))
        power_set = list(powerset(hospital_list))
        list_of_sets = []
        for curr_set in power_set:
            list_of_sets.append(set(curr_set))

        state_space = []
        for current_set in list_of_sets:
        # Handle empty set case
            if not current_set:
               state_space.append((0, current_set))
               for value in hospital_list:
                state_space.append((value, current_set))
            # If set has values, add tuples with not yet visited values
            else:
                not_yet_visited = hospital_set - current_set
                if not not_yet_visited:
                    state_space.append((-1, current_set)) #DONE
                else:
                    for curr_elem in not_yet_visited:
                        state_space.append((curr_elem, current_set))
        # print(state_space)
        # print(len(state_space))

        counter = 1
        for location, visited in state_space:
            self.state_to_index[(location, tuple(sorted(visited)))] = counter
            counter += 1

        # Generate dataset
        dataset = [] #(state, action, reward, next state)
        counter = 0
        for curr_state in state_space:
            counter += 1
            # print(counter)
            curr_location = curr_state[0]
            visited = curr_state[1].copy()
            if curr_location != 0:
                visited.add(curr_location)
            not_yet_visited = hospital_set - visited

            next_states = [(x, visited) for x in not_yet_visited]
            for next_state in next_states:
                action = next_state[0]
                curr_reward = self.reward(curr_state, next_state)

                curr_state_index = self.state_to_index[(curr_state[0], tuple(sorted(curr_state[1])))]
                next_state_index = self.state_to_index[(next_state[0], tuple(sorted(next_state[1])))]
                dataset.append([curr_state_index, action, curr_reward, next_state_index])

        # print(dataset)
        np.savetxt("random_with_urgency_preliminary_dataset.csv", dataset, fmt = '%1d,%1d,%1d,%1d')

        w = csv.writer(open("random_with_urgency_state_to_index.csv" , "w"))
        for key, value in self.state_to_index.items():
            w.writerow([tuple(key), value])

    ###################################################################################
    ############################### Q-LEARNING ########################################
    ###################################################################################
    def q_learning(self):
        def generate_policy(infile, outfile):
            dataset = list(csv.reader(open(infile)))
            dataset = np.array(dataset).astype(int)

            num_states = np.max(dataset[:, 3]) #Since we might have next_states that are never init_states
            print(num_states)
            num_actions = np.max(dataset[:, 1])
            print(num_actions)

            q_matrix = np.zeros((num_states, num_actions))
            starting_time = timeit.default_timer()
            for curr_iter in range(self.NUM_EPOCHS):
                print(curr_iter)
                counter = 0
                for curr_row in dataset:
                    counter += 1
                    #print(counter)
                    curr_state = curr_row[0]
                    curr_action = curr_row[1]
                    curr_reward = curr_row[2]
                    next_state = curr_row[3]

                    possible_next_q = []
                    for poss_action in range(num_actions):
                        possible_next_q.append(q_matrix[next_state - 1, poss_action]) #Keep same since we 1 index states
                    best_next_q = np.max(possible_next_q)

                    new_value = q_matrix[curr_state - 1][curr_action - 1] + self.lr * (curr_reward + (self.DISCOUNT_FACTOR * best_next_q) - q_matrix[curr_state - 1][curr_action - 1])
                    q_matrix[curr_state - 1][curr_action - 1] = new_value

            #For each state, output the best action
            best_action_list = []
            #print(q_matrix)
            with open("random_with_urgency_q_matrix.csv", 'w') as g:
                for curr_row in q_matrix:
                    g.write(str(curr_row) + '\n')

            for curr_row_index in range(num_states):
                q_matrix_row = q_matrix[curr_row_index]
                best_action = np.argmax(q_matrix_row) + 1

                current_state = curr_row_index + 1
                best_action_list.append((current_state, best_action))
            end_time = timeit.default_timer()

            with open(outfile, 'w') as f:
                for curr_value in best_action_list:
                    f.write(str(str(curr_value[0]) + "," + str(curr_value[1])) + "\n")
            print("Algorithm took:")
            print(end_time - starting_time)

        generate_policy(self.in_file, self.out_file)

    ###################################################################################
    ############################### GENERATE ROUTE ####################################
    ###################################################################################
    def generate_route(self):
        #TODO: Make this work for our new reward function
        def total_reward_for_route(route):
          total_reward = 0
          visited = set()
          for i in range(0, len(route)-1):
            visited_p = visited.copy()
            if route[i] != 0: 
                visited_p.add(route[i])
            total_reward += self.reward((route[i], sorted(visited)), (route[i+1], sorted(visited_p)))
            if route[i] != 0: 
                visited.add(route[i])

          return total_reward

        # From policy, output optimal route
        # Start from initial state, take best action, read next state, repeat.
        # At each step, write down the best action
        route = [0]  # start at home base

        # get policy from file
        policy = dict()  #  of length 5121
        with open(self.out_file) as file:
            # with open("improved_fixed_run.policy") as file:
            lines = file.readlines()
            for line in lines:
                line = line.rstrip()
                state, action = line.split(",")
                policy[int(state)] = int(action)

        # get optimal route
        curr_state_indx = 1
        history = set()
        prev_action = -1
        for i in range(self.NUM_HOSPITALS):
            if curr_state_indx != 1:  # if not in the start state
                history.add(prev_action)  # visit the next hospital

            best_action = policy[curr_state_indx]
            prev_action = best_action

            next_state = (best_action, tuple(sorted(history)))
            next_state_indx = self.state_to_index[next_state]
            curr_state_indx = next_state_indx
            route.append(best_action)  # update route

        print("route from q-learning: ", route)
        self.final_route = route

        total_reward = total_reward_for_route(route)
        print("total_reward from q-learning route: ", total_reward)

        our_route = [0, 5, 4, 8, 9, 6, 10, 1, 2, 7, 3]
        total_reward = total_reward_for_route(our_route)
        print("total_reward for our route: ", total_reward)

    def visualize_route(self):
        x = np.array([self.hospital_to_coord[self.final_route[i]][0] for i in range(len(self.final_route))])
        y = np.array([self.hospital_to_coord[self.final_route[i]][1] for i in range(len(self.final_route))])

        plt.clf()
        fig, ax = plt.subplots()
        plt.xlim([-1, 11])
        plt.ylim([-1, 11])
        plt.title("Drone Route Planning")
        plt.grid()

        arr_done = mpimg.imread('drone.png')
        imagebox = OffsetImage(arr_done, zoom=0.2)
        
        for i in range(0, len(self.final_route)):
            hospital_num = self.final_route[i]
            h_x, h_y = self.hospital_to_coord[hospital_num][0], self.hospital_to_coord[hospital_num][1] 
            plt.scatter(h_x, h_y, color='blue')
            if i != 0:
                plt.text(h_x+.1, h_y+.1, str(hospital_num), fontsize=9)
            else: # if plotting home base
                plt.text(h_x+.1, h_y+.1, "Home Base", fontsize=9)
        
        # show drone image at home base
        ab = AnnotationBbox(imagebox, (x[0], y[0]), frameon=0)
        ax.add_artist(ab)
        plt.pause(1)
        ab.remove()
        
        # add line segments for each pair of points, show drone at each step
        for i in range(0, len(self.final_route)-1):
            plt.plot(x[i:i+2], y[i:i+2], color='skyblue')
            ab = AnnotationBbox(imagebox, (x[i+1], y[i+1]), frameon=0)
            ax.add_artist(ab)
            plt.pause(1)
            ab.remove()
            #   plt.savefig(f"{i}.png")
        
        plt.savefig("latestDroneRoute.png")
        plt.grid()
        plt.show()

final_project = Final_Project()
final_project.generate_dataset()
final_project.q_learning()
final_project.generate_route()
final_project.visualize_route()