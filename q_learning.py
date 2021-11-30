import sys
import csv
import numpy as np
import timeit


NUM_EPOCHS = 100
lr = .1
DISCOUNT_FACTOR = .95 #.95 for small and large, 1 for medium

def generate_policy(infile, outfile):
    dataset = list(csv.reader(open(infile)))
    variables = dataset[0]
    del dataset[0]
    dataset = np.array(dataset).astype(int)

    num_states = np.max(dataset[:, 0])
    print(num_states)
    num_actions = np.max(dataset[:, 1])
    print(num_actions)

    q_matrix = np.zeros((num_states, num_actions))
    starting_time = timeit.default_timer()
    for curr_iter in range(NUM_EPOCHS):
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
                possible_next_q.append(q_matrix[next_state - 1, poss_action])
            best_next_q = np.max(possible_next_q)

            new_value = q_matrix[curr_state - 1][curr_action - 1] + lr * (curr_reward + (DISCOUNT_FACTOR * best_next_q) - q_matrix[curr_state - 1][curr_action - 1])
            q_matrix[curr_state - 1][curr_action - 1] = new_value

    #For each state, output the best action
    best_action_list = []
    #print(q_matrix)

    for curr_row_index in range(num_states):
        q_matrix_row = q_matrix[curr_row_index]
        best_action = np.argmax(q_matrix_row) + 1
        best_action_list.append(best_action)
    end_time = timeit.default_timer()

    with open(outfile, 'w') as f:
        for curr_value in best_action_list:
            f.write(str(curr_value) + "\n")
    print("Algorithm took:")
    print(end_time - starting_time)

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project2.py <infile>.csv <outfile>.policy")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    generate_policy(inputfilename, outputfilename)

if __name__ == '__main__':
    main()