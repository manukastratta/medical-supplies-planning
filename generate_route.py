POLICY_FILENAME = "policyWith100Epochs.policy"
NUM_HOSPITALS = 10

#TODO: Make this work for our new reward function
def total_reward_for_route(route, hospital_to_coord):
  total_reward = 0
  for i in range(0, len(route)-1):
    total_reward += reward(route[i], route[i+1], hospital_to_coord)
  return total_reward


# From policy, output optimal route
# Start from initial state, take best action, read next state, repeat.
# At each step, write down the best action
route = [0]  # start at home base

# get policy from file
policy = dict()  #  of length 5121
with open(POLICY_FILENAME) as file:
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
for i in range(NUM_HOSPITALS):
    if curr_state_indx != 1:  # if not in the start state
        history.add(prev_action)  # visit the next hospital

    best_action = policy[curr_state_indx]
    prev_action = best_action

    next_state = (best_action, tuple(sorted(history)))
    next_state_indx = state_to_index[next_state]
    curr_state_indx = next_state_indx
    route.append(best_action)  # update route

print("route from q-learning: ", route)

total_reward = total_reward_for_route(route, hospital_to_coord)
print("total_reward from q-learning route: ", total_reward)

our_route = [0, 5, 4, 8, 9, 6, 10, 1, 2, 7, 3]
total_reward = total_reward_for_route(our_route, hospital_to_coord)
print("total_reward for our route: ", total_reward)