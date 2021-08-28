#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sys
sys.path.insert(0, "../../scripts")
from utils import load_data, infer_L1000_features


# In[2]:


folder = 'latentSpaces'


# In[4]:


data_splits = ["complete"]
complete_df = load_data(data_splits, dataset='L1000')['complete']
meta_columns = infer_L1000_features(complete_df, metadata = True)


# In[6]:


vaes = ['','_mmd','_vanilla']


# In[7]:


for vae in vaes:
    latent_df = pd.read_csv(f'latentTwoLayer{vae}.csv').drop('Unnamed: 0', axis = 1)
    combined_df = pd.concat([complete_df[meta_columns], latent_df], axis = 1)
    if vae == '':
        vae = '_beta'
    combined_df.to_csv(folder + f'/L1000Latent{vae}_metadata.csv', index = False)

