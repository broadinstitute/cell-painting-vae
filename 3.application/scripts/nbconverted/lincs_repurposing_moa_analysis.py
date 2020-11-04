#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import os
from pycytominer.cyto_utils import infer_cp_features
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, "../scripts")
from utils import load_data


# In[9]:


data_dict = load_data(["complete"])
meta_features = infer_cp_features(data_dict["complete"], metadata=True)
cp_features = infer_cp_features(data_dict["complete"])

complete_features_df = data_dict["complete"].reindex(cp_features, axis="columns")
complete_meta_df = data_dict["complete"].reindex(meta_features, axis="columns")


# In[10]:


#obtain only the moa column from the whole dataframe
df = pd.read_csv("repurposing_info_external_moa_map_resolved.tsv",sep='\t').set_index('broad_sample').reindex(index=complete_meta_df['Metadata_broad_sample']).reset_index().drop('Metadata_broad_sample',axis = 1)
moa = df.moa.dropna()
moa


# In[11]:


pipes = moa[moa.str.contains("\|")]
unique_pipes = pipes.drop_duplicates()
no_pipes = moa[~moa.str.contains("\|")]


# In[12]:


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


# In[14]:


moas_occurrence_df = pd.concat([unique_pipes, moas_occurrences_all_pipes_df],axis = 1)
moas_occurrence_df = moas_occurrence_df.sort_values("full moa occurrence", ascending = False)
moas_occurrence_df.index = moas_occurrence_df['moa']
moas_occurrence_df = moas_occurrence_df.drop('moa',axis =1)
moas_occurrence_df.to_csv("moas_occurrence.tsv", sep = "\t")
moas_occurrence_df.head(10)


# In[ ]:




