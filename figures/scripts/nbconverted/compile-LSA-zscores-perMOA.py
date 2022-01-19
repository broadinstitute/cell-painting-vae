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
cp_l2_file = pathlib.Path("..", "cell-painting", "3.application", "L2_distances_with_moas.csv")
cp_l2_df = pd.read_csv(cp_l2_file).assign(shuffled="real")

cp_l2_df.loc[cp_l2_df.Model.str.contains("Shuffled"), "shuffled"] = "shuffled"
cp_l2_df = cp_l2_df.assign(
    architecture=[x[-1] for x in cp_l2_df.Model.str.split(" ")],
    assay="CellPainting",
    metric="L2 distance"
).rename(columns={"L2 Distance": "metric_value"})

print(cp_l2_df.shape)
cp_l2_df.head()


# In[3]:


# Load Pearson correlations per MOA
cp_file = pathlib.Path("..", "cell-painting", "3.application", "pearson_with_moas.csv")
cp_pearson_df = pd.read_csv(cp_file).assign(shuffled="real")

cp_pearson_df.loc[cp_pearson_df.Model.str.contains("Shuffled"), "shuffled"] = "shuffled"
cp_pearson_df = cp_pearson_df.assign(
    architecture=[x[-1] for x in cp_pearson_df.Model.str.split(" ")],
    assay="CellPainting",
    metric="Pearson correlation"
).rename(columns={"Pearson": "metric_value"})

print(cp_pearson_df.shape)
cp_pearson_df.head()


# In[4]:


# Combine data
cp_df = pd.concat([cp_l2_df, cp_pearson_df]).reset_index(drop=True)

print(cp_df.shape)
cp_df.head()


# In[5]:


all_moas = cp_df.MOA.unique().tolist()
print(len(all_moas))
all_metrics = cp_df.metric.unique().tolist()
all_architectures = cp_df.architecture.unique().tolist()
all_architectures


# In[6]:


results_df = []
for metric in all_metrics:
    for moa in all_moas:
        for arch in all_architectures:
            # subset data to include moa per architecture
            sub_cp_df = (
                cp_df
                .query(f"metric == '{metric}'")
                .query(f"architecture == '{arch}'")
                .query(f"MOA == '{moa}'")
                .reset_index(drop=True)
            )

            real_ = sub_cp_df.query("shuffled == 'real'").loc[:, "metric_value"].tolist()
            shuff_ = sub_cp_df.query("shuffled != 'real'").loc[:, "metric_value"].tolist()

            # Calculate zscore consistently with other experiments
            zscore_result = scipy.stats.zscore(shuff_ + real_)[-1]
            results_df.append([moa, arch, zscore_result, metric])

# Compile results
results_df = pd.DataFrame(results_df, columns=["MOA", "model", "zscore", "metric"])

print(results_df.shape)
results_df.head()


# In[7]:


# Output data
output_file = pathlib.Path("data", "MOA_LSA_metrics.tsv")
results_df.to_csv(output_file, sep="\t", index=False)

