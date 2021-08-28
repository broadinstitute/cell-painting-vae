#!/usr/bin/env python
# coding: utf-8

# # Split the L1000 Data into Training/Testing/Validation Sets
# 
# Split the data 80% training, 10% testing, 10% validation, balanced by platemap.

# In[1]:


import sys
import pathlib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from pycytominer import feature_select
from pycytominer.cyto_utils import infer_cp_features

sys.path.insert(0, "../../scripts")
from utils import transform, infer_L1000_features


# In[2]:


# %load_ext nb_black


# In[3]:


seed = 9876
test_split = 0.2

output_dir = pathlib.Path("data")
output_dir.mkdir(exist_ok=True)


# In[4]:


# Load data
phase2_L1000_df = pd.read_csv("../0B.process-data/data/L1000_phase2.tsv.gz", sep="\t")

print(phase2_L1000_df.shape)
phase2_L1000_df.head(2)


# In[5]:


# phase2_L1000_df['pert_id_cell_id'] = phase2_L1000_df['pert_id'] + '_' + phase2_L1000_df['cell_id']


# In[6]:


# pert_id_cell_id = np.array(phase2_L1000_df.pert_id_cell_id)
# unique, counts = np.unique(pert_id_cell_id, return_counts=True)
# count_of_pert_id_cell_id = dict(zip(unique, counts))


# In[7]:


# valid_pert_id_cell_id = []
# for key in count_of_pert_id_cell_id:
#     if count_of_pert_id_cell_id[key] > 2:
#         valid_pert_id_cell_id.append(key)


# In[8]:


# phase2_L1000_df = phase2_L1000_df[phase2_L1000_df.pert_id_cell_id.isin(valid_pert_id_cell_id)]


# In[9]:


features = infer_L1000_features(phase2_L1000_df)
meta_features = infer_L1000_features(phase2_L1000_df, metadata=True)


# In[10]:


# Zero One Normalize Data
# phase2_L1000_df = transform(
#     phase2_L1000_df, features=features, meta_features=meta_features, operation = "-1+1"
# )


# In[11]:


# Split data into 80% train, 20% test
train_df, test_df = train_test_split(
    phase2_L1000_df,
    test_size=test_split,
    random_state=seed,
    stratify=phase2_L1000_df.cell_id,
)


# In[12]:


# Split test data into 50% validation, 50% test
test_df, valid_df = train_test_split(
    test_df,
    test_size=0.5,
    random_state=seed,
    stratify=test_df.cell_id,
)


# In[13]:


print(train_df.shape)
print(test_df.shape)
print(valid_df.shape)


# In[14]:


# Output data splits
train_file = pathlib.Path(output_dir, "L1000PHASE2noscaling_train.tsv.gz")
test_file = pathlib.Path(output_dir, "L1000PHASE2noscaling_test.tsv.gz")
valid_file = pathlib.Path(output_dir, "L1000PHASE2noscaling_valid.tsv.gz")
complete_file = pathlib.Path(output_dir, "L1000PHASE2noscaling_complete.tsv.gz")

# train_df.to_csv(train_file, sep="\t", index=False, float_format="%.5g")
# test_df.to_csv(test_file, sep="\t", index=False, float_format="%.5g")
# valid_df.to_csv(valid_file, sep="\t", index=False, float_format="%.5g")
phase2_L1000_df.to_csv(complete_file, sep="\t", index=False, float_format="%.5g")

