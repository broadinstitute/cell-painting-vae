#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, "../../scripts")
from utils import load_data, infer_L1000_features


# In[2]:


data_dict = load_data(["test"], dataset = 'L1000')
# complete_features_df = data_dict["complete"]
# data_dict["complete"] = data_dict["complete"][data_dict["complete"].cell_id == "A549"].reset_index(drop = True)
# data_dict["complete"] = data_dict["complete"]


meta_features = infer_L1000_features(data_dict["test"], metadata=True)
cp_features = infer_L1000_features(data_dict["test"])

complete_features_df = data_dict["test"].reindex(cp_features, axis="columns")
complete_meta_df = data_dict["test"].reindex(meta_features, axis="columns")


# In[3]:


# input_dir = '../0.download-data/data/'
# complete_meta_df = pd.read_csv(input_dir + "col_meta_level_3_REP.A_A549_only_n27837.txt", sep = "\t")


# In[4]:


pd.read_csv("repurposing_info_external_moa_map_resolved.tsv",sep='\t')


# In[5]:


#obtain only the moa column from the whole dataframe
df = pd.read_csv("repurposing_info_external_moa_map_resolved.tsv",sep='\t').set_index('broad_id').reindex(index=complete_meta_df['pert_id']).reset_index().drop('pert_id',axis = 1)
moa = df.moa.dropna()
moa


# In[6]:


pipes = moa[moa.str.contains("\|")]
unique_pipes = pipes.drop_duplicates()
no_pipes = moa[~moa.str.contains("\|")]


# In[7]:


#count the number of occurrences of each moa in each moa combination
moas_occurences_all_pipes = []
for pipe in unique_pipes:
    moas = pipe.split("|")
    moas_occurences = [pipes[pipes == pipe].count()]
    for moa in moas:
        moas_occurences.append(no_pipes[no_pipes == moa].count())
    moas_occurences_all_pipes.append(moas_occurences)
moas_occurrences_all_pipes_df = pd.DataFrame(data=moas_occurences_all_pipes, index = unique_pipes.index.values.tolist(), columns=["full moa occurrence","moa1 occurrence", "moa2 occurrence","moa3 occurrence", "moa4 occurrence", "moa5 occurrence", "moa6 occurrence"])
moas_occurrences_all_pipes_df


# In[8]:


moas_occurrence_df = pd.concat([unique_pipes, moas_occurrences_all_pipes_df],axis = 1)
moas_occurrence_df = moas_occurrence_df.sort_values("full moa occurrence", ascending = False)
moas_occurrence_df.index = moas_occurrence_df['moa']
moas_occurrence_df = moas_occurrence_df.drop('moa',axis =1)
moas_occurrence_df.to_csv("moas_occurrence.tsv", sep = "\t")
moas_occurrence_df.head(10)


# In[9]:


moas_occurrence_df

