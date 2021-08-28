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


features = infer_L1000_features(phase2_L1000_df)
meta_features = infer_L1000_features(phase2_L1000_df, metadata=True)


# In[6]:


# Zero One Normalize Data
phase2_L1000_df = transform(
    phase2_L1000_df, features=features, meta_features=meta_features, operation = "-1+1"
)


# In[7]:


phase2_L1000_df


# In[8]:


# Split data into 80% train, 20% test
train_df, test_df = train_test_split(
    phase2_L1000_df,
    test_size=test_split,
    random_state=seed,
    stratify=phase2_L1000_df.cell_id,
)


# In[9]:


# Split test data into 50% validation, 50% test
test_df, valid_df = train_test_split(
    test_df,
    test_size=0.5,
    random_state=seed,
    stratify=test_df.cell_id,
)


# In[10]:


print(train_df.shape)
print(test_df.shape)
print(valid_df.shape)


# In[11]:


# Output data splits
train_file = pathlib.Path(output_dir, "L1000PHASE2level4-1+1_train.tsv.gz")
test_file = pathlib.Path(output_dir, "L1000PHASE2level4-1+1_test.tsv.gz")
valid_file = pathlib.Path(output_dir, "L1000PHASE2level4-1+1_valid.tsv.gz")
complete_file = pathlib.Path(output_dir, "L1000PHASE2level4-1+1_complete.tsv.gz")

# train_df.to_csv(train_file, sep="\t", index=False, float_format="%.5g")
# test_df.to_csv(test_file, sep="\t", index=False, float_format="%.5g")
# valid_df.to_csv(valid_file, sep="\t", index=False, float_format="%.5g")
phase2_L1000_df.to_csv(complete_file, sep="\t", index=False, float_format="%.5g")

