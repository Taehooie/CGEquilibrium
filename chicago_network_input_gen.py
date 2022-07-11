import pandas as pd
import numpy as np
import sys
import jax.numpy as jnp

# Data selection for a pair of OD
"""Data Load"""
agent1 = pd.read_csv('agent_type_1_v2.csv')
agent2 = pd.read_csv('agent_type_2_v2.csv')
agent3 = pd.read_csv('agent_type_3_v2.csv')
agent4 = pd.read_csv('agent_type_4_v2.csv')

# Origin Layer
ozone_df = agent1
ozone_df.rename(columns={'agent_id':'ozone_id'}, inplace=True)

# Origin - Destination
od_df = agent2
od_df.rename(columns={'agent_id':'od_id'}, inplace=True)

# Path
path_df = agent3
path_df.rename(columns={'agent_id':'path_id'}, inplace=True)

# Link
link_df = agent4
link_df.rename(columns={'agent_id':'link_id'}, inplace=True)

# Reset index of each file
ozone_df.reset_index(drop=True, inplace=True)
od_df.reset_index(drop=True, inplace=True)
path_df.reset_index(drop=True, inplace=True)
link_df.reset_index(drop=True, inplace=True)

"""Set up lengths of each layer"""
num_ozone = ozone_df.shape[0]
num_od = od_df.shape[0]
num_path = path_df.shape[0]
num_link = link_df.shape[0]

"""Define dictionaries"""
# Origin layer
node_zone_dict = ozone_df[['o_node_id', 'ozone_id']].set_index('o_node_id').to_dict()['ozone_id']

# Origin-Destination layer
od_df['od_pair'] = od_df.apply(lambda x: (int(x.o_zone_id), int(x.d_zone_id)), axis=1)
od_df['ozone_id'] = od_df.apply(lambda x: node_zone_dict[int(x.o_zone_id)], axis=1)
od_pair_dict = od_df[['od_pair', 'od_id']].set_index('od_pair').to_dict()['od_id']

# Path to Origin-Destination
path_df['od_id'] = path_df.apply(lambda x: od_pair_dict[int(x.o_zone_id), int(x.d_zone_id)], axis=1)
path_od_dict = path_df[['path_id', 'od_id']].set_index('path_id').to_dict()['od_id']

# Link layer
link_df['link_pair'] = link_df.apply(lambda x: (int(x.from_node_id), int(x.to_node_id)), axis=1)  
link_id_pair_dict = link_df[['link_id', 'link_pair']].set_index('link_pair').to_dict()['link_id']

# Origin-Destination to Origin
od_ozone_dict = od_df[['ozone_id', 'od_id']].set_index('od_id').to_dict()['ozone_id']

# 1. Generate the orgin-destination volume
od_volumn = jnp.array(agent2['od_demand'])
od_volumn = od_volumn.reshape(od_volumn.shape[0], 1)

# 2. Generate the od-path incidence matrix
# To count the number of x_f flow variables
count_x_f = []
for i in path_df['o_node_id'].unique():
    
    for j in path_df['d_node_id'].unique():
        
        index = path_df[(path_df['o_node_id'] == i) & (path_df['d_node_id'] == j)].index
        
        if len(index)>0:
            
            for k in range(len(index)):
                
                if k==len(index)-1: 
                    
                    count_x_f.append(k)

# the size of the column
num_val = np.array(count_x_f).sum()
od_path_inc = np.zeros([od_volumn.shape[0], num_val], dtype=np.float64)
val_loc = np.where(np.array(count_x_f)>0)

start_index = 0
for i in val_loc[0]:

  od_path_inc[i][start_index: start_index+ count_x_f[i]] = 1

  start_index = start_index + count_x_f[i]

# 3. Generate the path-link incidence matrix
path_link_inc_mat = np.zeros([num_path, num_link], dtype=np.float64)
for i in range(num_path):
    path_r = path_df.loc[i]
    node_list = path_r.node_sequence.split(';')
    for link_l in range(len(node_list) - 1):
        link_pair = (int(node_list[len(node_list) - 1 - link_l]), int(node_list[len(node_list) - 2 - link_l]))
        link_id = link_id_pair_dict[link_pair]
        path_link_inc_mat[int(path_r.path_id - 1)][int(link_id - 1)] = 1.0

# To find the path flow index 
path_link_index = []
path_link_index2 = []
for i in path_df['o_node_id'].unique():
    
    for j in  np.sort(path_df['d_node_id'].unique()):
        
        index = np.array(path_df[(path_df['o_node_id'] == i) & (path_df['d_node_id'] == j)].index)

        if len(index)==1:
          path_link_index2.append(index)

        if len(index)>1:
          n_1_index = index[0 : -1]
          path_link_index.append(n_1_index)

          path_link_index2.append(index[-1:])

index_1 = np.concatenate(path_link_index)
index_2 = np.concatenate(path_link_index2)

path_link_inc = path_link_inc_mat[index_1]
path_link_inc_n = path_link_inc_mat[index_2]

# Transit Costs
transit_cost = jnp.array(od_df['od_time'] * 1.2)

# Link Features (free flow travel time (fftt) and capacity (cap))
fftt = jnp.array(link_df['VDF_fftt1'])
cap  = jnp.array(link_df['VDF_cap1'])

od_path_inc_for_min = (jnp.ones(jnp.shape(od_path_inc)) - od_path_inc) * 1e5 + od_path_inc