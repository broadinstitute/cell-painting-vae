#!/usr/bin/env python
# coding: utf-8

# ## Compile per MOA p value for shuffled comparison

# In[1]:


import pathlib
import numpy as np
import pandas as pd
import scipy.stats


# In[2]:


# Load L2 distances per MOA
cp_file = pathlib.Path("..", "cell-painting", "3.application", "L2_distances_with_moas.csv")
cp_df = pd.read_csv(cp_file).assign(shuffled="real")

cp_df.loc[cp_df.Model.str.contains("Shuffled"), "shuffled"] = "shuffled"
cp_df = cp_df.assign(
    architecture=[x[-1] for x in cp_df.Model.str.split(" ")],
    assay="CellPainting"
)

print(cp_df.shape)
cp_df.head()


# In[3]:


all_moas = cp_df.MOA.unique().tolist()
print(len(all_moas))
all_architectures = cp_df.architecture.unique().tolist()
all_architectures


# In[4]:


results_df = []
for moa in all_moas:
    for arch in all_architectures:
        # subset data to include moa per architecture
        sub_cp_df = (
            cp_df
            .query(f"architecture == '{arch}'")
            .query(f"MOA == '{moa}'")
            .reset_index(drop=True)
        )
        
        real_ = sub_cp_df.query("shuffled == 'real'").loc[:, "L2 Distance"].tolist()
        shuff_ = sub_cp_df.query("shuffled != 'real'").loc[:, "L2 Distance"].tolist()
        
        # Calculate zscore consistently with other experiments
        zscore_result = scipy.stats.zscore(shuff_ + real_)[-1]
        results_df.append([moa, arch, zscore_result])

# Compile results
results_df = pd.DataFrame(results_df, columns=["MOA", "model", "zscore"])

print(results_df.shape)
results_df.head()


# In[5]:


# Output data
output_file = pathlib.Path("data", "MOA_LSA_zscores.tsv")
results_df.to_csv(output_file, sep="\t", index=False)

