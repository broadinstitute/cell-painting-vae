#!/usr/bin/env python
# coding: utf-8

# # Split the Cell Painting Data into Training/Testing/Validation Sets
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
from utils import transform


# In[2]:


# %load_ext nb_black


# In[3]:


seed = 9876
test_split = 0.2
feature_select_opts = [
    "variance_threshold",
    "blacklist",
    "drop_na_columns",
#     "drop_outliers",
]

output_dir = pathlib.Path("data")
output_dir.mkdir(exist_ok=True)


# In[4]:


consensus = "modz"
batch = "2016_04_01_a549_48hr_batch1"
commit_hash = "27a2d7dd74067b5754c2c045e9b1a9cfb0581ae4"

# We have noticed particular technical issues with this platemap
# remove it from downstream consideration
# https://github.com/broadinstitute/lincs-cell-painting/issues/43
filter_platemap = "C-7161-01-LM6-011"


# In[5]:


# Load data
base_url = (
    "https://media.githubusercontent.com/media/broadinstitute/lincs-cell-painting/"
)
# repurp_url = (
#     f"{base_url}/{commit_hash}/consensus/{batch}/{batch}_consensus_{consensus}.csv.gz"
# )
repurp_url = 'https://github.com/broadinstitute/lincs-cell-painting/raw/master/spherized_profiles/profiles/2016_04_01_a549_48hr_batch1_dmso_spherized_profiles_with_input_normalized_by_whole_plate.csv.gz'

complete_level4_df = pd.read_csv(repurp_url).query(
    "Metadata_Plate_Map_Name != @filter_platemap"
)

# complete_consensus_df = complete_consensus_df.assign(
#     Metadata_unique_id=complete_consensus_df.Metadata_broad_sample
#     + "_dose_"
#     + complete_consensus_df.Metadata_dose_recode.astype(str)
# )

print(complete_level4_df.shape)
complete_level4_df.head(2)


# In[6]:


# Perform feature selection
complete_level4_df = feature_select(
    profiles=complete_level4_df,
    features="infer",
    samples="none",
    operation=feature_select_opts,
    output_file="none",
    na_cutoff=0,
    corr_threshold=0.9,
    corr_method="pearson",
    freq_cut=0.05,
    unique_cut=0.1,
)

print(complete_level4_df.shape)


# In[15]:


# Zero One Normalize Data
complete_level4_df = transform(complete_level4_df)


# In[10]:


# Split data
train_df, test_df = train_test_split(
    complete_level4_df,
    test_size=test_split,
    random_state=seed,
    stratify=complete_level4_df.Metadata_Plate_Map_Name,
)

test_df, valid_df = train_test_split(
    test_df,
    test_size=0.5,
    random_state=seed,
    stratify=test_df.Metadata_Plate_Map_Name,
)


# In[11]:


print(train_df.shape)
print(test_df.shape)
print(valid_df.shape)


# In[13]:


# Output data splits
train_file = pathlib.Path(output_dir, "cell_painting_trainLEVEL4.tsv.gz")
test_file = pathlib.Path(output_dir, "cell_painting_testLEVEL4.tsv.gz")
valid_file = pathlib.Path(output_dir, "cell_painting_validLEVEL4.tsv.gz")
complete_file = pathlib.Path(output_dir, "cell_painting_completeLEVEL4.tsv.gz")

# train_df.to_csv(train_file, sep="\t", index=False, float_format="%.5g")
# test_df.to_csv(test_file, sep="\t", index=False, float_format="%.5g")
# valid_df.to_csv(valid_file, sep="\t", index=False, float_format="%.5g")
complete_level4_df.to_csv(complete_file, sep="\t", index=False, float_format="%.5g")

