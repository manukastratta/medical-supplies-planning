import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import random
from itertools import chain, combinations
import csv


def powerset(s):
  return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))




# # Generate true distributions of weekly hospital orders

NUM_HOSPITALS = 10
N_SAMPLES = 5
random.seed(50)

# https://towardsdatascience.com/exploring-normal-distribution-with-jupyter-notebook-3645ec2d83f8

blood_distrs = list()
vaccine_distrs = list()

# denotes the x-axis range
blood_data = np.arange(0, 600, 0.01)
vaccine_data = np.arange(0, 900, 0.01)

fig_blood, ax_blood = plt.subplots()
fig_vaccine, ax_vaccine = plt.subplots()

for i in range(1, NUM_HOSPITALS+1):
  mu_blood, sigma_blood = i*50, random.uniform(3*i, 5*i)
  mu_vaccine, sigma_vaccine = i*100, random.uniform(6*i, 10*i)

  print(f'Blood -> Hospital {i}: {mu_blood}, {sigma_blood}')
  print(f'Vaccine -> Hospital {i}: {mu_vaccine}, {sigma_vaccine}')
  blood_samples = np.random.normal(mu_blood, sigma_blood, N_SAMPLES)
  vaccine_samples = np.random.normal(mu_vaccine, sigma_vaccine, N_SAMPLES)

  blood_distrs.append(np.rint(blood_samples)) # round to nearest integer
  vaccine_distrs.append(np.rint(vaccine_samples))

  ax_blood.plot(blood_data, norm.pdf(blood_data, scale=sigma_blood, loc=mu_blood), label=f"Hospital #{i}")
  ax_vaccine.plot(vaccine_data, norm.pdf(vaccine_data, scale=sigma_vaccine, loc=mu_vaccine), label=f"Hospital #{i}")

ax_blood.set_title('True Distribution of Blood Supply Orders')
ax_blood.legend(loc='best', frameon=True)

ax_vaccine.set_title('True Distribution of Vaccine Supply Orders')
ax_vaccine.legend(loc='best', frameon=True)

# Save hospital past order data, 1 file per hospital
for i in range(0, NUM_HOSPITALS):
  np.savetxt(f'blood_data/hospital{i+1}.txt', blood_distrs[i])
  np.savetxt(f'vaccine_data/hospital{i+1}.txt', vaccine_distrs[i])

# For every hospital, learn a distribution
hospital_to_blood_dist = dict()
hospital_to_vaccine_dist = dict()
for i in range(0, NUM_HOSPITALS):
  blood_mean, blood_std = norm.fit(blood_distrs[i][:])
  hospital_to_blood_dist[i+1] = (blood_mean, blood_std)

  vaccine_mean, vaccine_std = norm.fit(vaccine_distrs[i][:])
  hospital_to_vaccine_dist[i+1] = (vaccine_mean, vaccine_std)

print(f'hospital_to_blood_dist: {hospital_to_blood_dist}')
print(f'hospital_to_vaccine_dist: {hospital_to_vaccine_dist}')


# Generate grid, hospital locations, and dataset: 

#Generate random hospital locations:
# MAX_DIM = 10
# possible_points = []
# for x in range(0, MAX_DIM + 1):
#   for y in range(0, MAX_DIM + 1):
#     curr_tuple = (x,y)
#     possible_points.append(curr_tuple)
# hospital_coords = random.sample(possible_points, NUM_HOSPITALS)
# print(hospital_coords)
#
# index_to_hospital = {} #hospitals start at 1, 0 refers to starting location [states list]
# index_to_hospital[0] = (0,0)
# for curr_index in range(1, NUM_HOSPITALS + 1):
#   index_to_hospital[curr_index] = hospital_coords[curr_index - 1]
# print(index_to_hospital)

#Generate fixed hospital locations:
# fixed_hospital_coords = [(x, x) for x in range(1, NUM_HOSPITALS + 1)]
# index_to_hospital = {} #hospitals start at 1, 0 refers to starting location [states list]
# index_to_hospital[0] = (0,0)
# for curr_index in range(1, NUM_HOSPITALS + 1): 
#   index_to_hospital[curr_index] = fixed_hospital_coords[curr_index - 1]
# print(index_to_hospital)

#Old state generation code:
# for curr_state in range(0, NUM_HOSPITALS + 1):
#   next_states = [x for x in state_list if x != curr_state]
#   for next_state in next_states:
#     action = next_state
#     curr_location = np.array(index_to_hospital[curr_state])
#     next_location = np.array(index_to_hospital[next_state])
#     curr_reward = int(25 / np.linalg.norm(curr_location - next_location)) #Just based on distance for now
#     dataset.append([curr_state, action, curr_reward, next_state])

#Generate random hospital locations:
MAX_DIM = 10
possible_points = []
for x in range(0, MAX_DIM + 1):
  for y in range(0, MAX_DIM + 1):
    curr_tuple = (x,y)
    possible_points.append(curr_tuple)
hospital_coords = random.sample(possible_points, NUM_HOSPITALS)

index_to_hospital = {} #hospitals start at 1, 0 refers to starting location [states list]
index_to_hospital[0] = (0,0)
for curr_index in range(1, NUM_HOSPITALS + 1): 
  index_to_hospital[curr_index] = hospital_coords[curr_index - 1]
print(index_to_hospital)

#state = (curr_state, set of states visited already)
#Generate powerset / build up state space
hospital_list = list(range(1, NUM_HOSPITALS + 1))
hospital_set = set(range(1, NUM_HOSPITALS + 1))
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
  #If set has values, add tuples with not yet visited values
  else:
    not_yet_visited = hospital_set - current_set
    if not not_yet_visited:
      state_space.append((-1, current_set)) #DONE
    else:
      for curr_elem in not_yet_visited:
        state_space.append((curr_elem, current_set))
# print(state_space)
# print(len(state_space))

state_to_index = {}
counter = 1
for location, visited in state_space:
  state_to_index[(location, tuple(sorted(visited)))] = counter
  counter += 1

#Generate dataset
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
  #print(not_yet_visited)
  next_states = [(x, visited) for x in not_yet_visited]
  for next_state in next_states:
    action = next_state[0]
    curr_grid_location = np.array(index_to_hospital[curr_location])
    next_grid_location = np.array(index_to_hospital[action])
    curr_reward = int(25 / np.linalg.norm (curr_grid_location - next_grid_location))
    # print(curr_state)
    curr_state_index = state_to_index[(curr_state[0], tuple(sorted(curr_state[1])))]
    next_state_index = state_to_index[(next_state[0], tuple(sorted(next_state[1])))]
    dataset.append([curr_state_index, action, curr_reward, next_state_index])

# print(dataset)
np.savetxt("fixed_preliminary_dataset.csv", dataset, fmt = '%1d,%1d,%1d,%1d')

w = csv.writer(open("state_to_index.csv" , "w"))
for key, value in state_to_index.items():
  w.writerow([tuple(key), value])


# From policy, output optimal route
# Start from initial state, take best action, read next state, repeat. 
# At each step, write down the best action
route = []

# get policy from file
policy = dict() # of length 5121
with open("my_test_policy.policy") as file:
    lines = file.readlines()
    for line in lines:
        line = line.rstrip()
        state, action = line.split(",")
        policy[int(state)] = int(action)

# get optimal route
curr_state_indx = 1
history = set()
prev_action = -1
for i in range(NUM_HOSPITALS):
  if curr_state_indx != 1: # if not in the start state
    history.add(prev_action) # visit the next hospital
  
  best_action = policy[curr_state_indx]
  prev_action = best_action

  next_state = (best_action, tuple(history))
  next_state_indx = state_to_index[next_state]
  curr_state_indx = next_state_indx
  route.append(best_action) # update route

print("route: ", route)