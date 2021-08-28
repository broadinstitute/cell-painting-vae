#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sys
sys.path.insert(0, "../../scripts")
from utils import load_data
from pycytominer.cyto_utils import infer_cp_features


# In[2]:


folder = 'latentSpaces'


# In[3]:


data_splits = ["complete"]
complete_df = load_data(data_splits)['complete']
meta_columns = infer_cp_features(complete_df, metadata = True)


# In[4]:


vaes = ['','_mmd','_vanilla']


# In[5]:


for vae in vaes:
    latent_df = pd.read_csv(f'level4Latent{vae}.csv').drop('Unnamed: 0', axis = 1)
    combined_df = pd.concat([complete_df[meta_columns], latent_df], axis = 1)
    if vae == '':
        vae = '_beta'
    combined_df.to_csv(folder + f'/level4Latent{vae}_metadata.csv', index = False)


# In[6]:


pd.read_csv('level4Latent_vanilla.csv')


# In[ ]:




