#!/usr/bin/env python
# coding: utf-8

# In[6]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import random


# # Generate true distributions of weekly hospital orders

# In[ ]:


NUM_HOSPITALS = 10
N_SAMPLES = 1000
random.seed(50)


# In[ ]:


# https://towardsdatascience.com/exploring-normal-distribution-with-jupyter-notebook-3645ec2d83f8

blood_distrs = list()
vaccine_distrs = list()

# denotes the x-axis range
blood_data = np.arange(0, 600, 0.01)
vaccine_data = np.arange(0, 900, 0.01)

fig_blood, ax_blood = plt.subplots()
fig_vaccine, ax_vaccine = plt.subplots()

for i in range(1, NUM_HOSPITALS+1):
  mu_blood, sigma_blood = i*50, random.uniform(1, 25)
  mu_vaccine, sigma_vaccine = i*75, random.uniform(1, 25)

  print(f'Hospital {i}: {mu_blood}, {sigma_blood}')
  blood_distrs.append(np.random.normal(mu_blood, sigma_blood, N_SAMPLES))
  vaccine_distrs.append(np.random.normal(mu_vaccine, sigma_vaccine, N_SAMPLES))

  ax_blood.plot(blood_data, norm.pdf(blood_data, scale=sigma_blood, loc=mu_blood), label=f"Hospital #{i}")
  ax_vaccine.plot(vaccine_data, norm.pdf(vaccine_data, scale=sigma_vaccine, loc=mu_vaccine), label=f"Hospital #{i}")

ax_blood.set_title('True Distribution of Blood Supply Orders')
ax_blood.legend(loc='best', frameon=True)

ax_vaccine.set_title('True Distribution of Vaccine Supply Orders')
ax_vaccine.legend(loc='best', frameon=True)


# In[ ]:





# In[ ]:


# creates 5 chunks-worth of data
# simply takes chunks of 200 entries from our normal distribution
# to view files, click on the left pane and click on the "Folder" icon at the bottom
for i in range(5):
  np.savetxt(f'{5-i}_weeks_ago.out', blood_distrs[0][200*i:200*(i+1)])


# creates an actual file in Google Drive
# alternate technique of saving to a file
# with open('/content/drive/My Drive/test.txt', 'w') as f:
#   f.write(np.array2string(blood_distrs[0][:200], separator='\n'))


# In[ ]:


# built-in function that allows us to find mean given a list of numbers
# (if we don't go with the Beta/ Dirichlet distribution)
mean, std = norm.fit(blood_distrs[0][200:400])


# Generate grid, hospital locations, and dataset: 

# In[ ]:


MAX_DIM = 10
possible_points = [] 
for x in range(0, MAX_DIM + 1): 
  for y in range(0, MAX_DIM + 1): 
    curr_tuple = (x,y)
    possible_points.append(curr_tuple)
hospital_coords = random.sample(possible_points, NUM_HOSPITALS)
print(hospital_coords)

index_to_hospital = {} #hospitals start at 1, 0 refers to starting location [states list]
index_to_hospital[0] = (0,0)
for curr_index in range(1, NUM_HOSPITALS + 1): 
  index_to_hospital[curr_index] = hospital_coords[curr_index - 1]
print(index_to_hospital)

dataset = [] #(state, action, reward, next state)
state_list = list(range(0, NUM_HOSPITALS + 1))
print(state_list)
for curr_state in range(0, NUM_HOSPITALS + 1): 
  next_states = [x for x in state_list if x != curr_state]
  action = 1 #1-indexed like project 
  for next_state in next_states: 
    curr_location = np.array(index_to_hospital[curr_state])
    next_location = np.array(index_to_hospital[next_state])
    curr_reward = 1 / np.linalg.norm(curr_location - next_location) #Just based on distance for now 
    dataset.append([curr_state, action, curr_reward, next_state])
    action += 1 
print(dataset)

