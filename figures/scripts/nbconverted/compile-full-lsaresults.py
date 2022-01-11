#!/usr/bin/env python
# coding: utf-8

# ## Compile Polypharmacology LSA results

# In[1]:


import pathlib
import pandas as pd


# In[14]:


assays = ["L1000", "cell-painting"]
models = ["beta", "vanilla", "mmd"]

analysis_dir = "3.application"

lsa_dfs = []
for assay in assays:
    data_dir = pathlib.Path(f"../{assay}/{analysis_dir}")
    lsa_results_files = [x for x in data_dir.iterdir() if "_general" in x.name]
    for lsa_results_file in lsa_results_files:
        
        # Extract info from filename
        file_info = lsa_results_file.name.split("_")
        if assay == "L1000":
            data_level = "level5"
            try:
                model = file_info[2].replace(".tsv", "")
            except IndexError:
                model = "beta"
        else:
            data_level = file_info[2].replace(".tsv", "")
            try:
                model = file_info[3].replace(".tsv", "")
            except IndexError:
                model = "beta"
        
        # Read data and process
        lsa_df = pd.read_csv(lsa_results_file, index_col=0, sep="\t")
        
        lsa_melt_df = (
            lsa_df
            .melt(var_name="input_data_type_full", value_name="dist")
        )
        id_df = (
            pd.DataFrame.from_records(
                lsa_melt_df.input_data_type_full.str.split(" "),
                columns = ["input_data_type", "shuffled"]
            )
        )
        
        lsa_melt_df = (
            pd.concat([lsa_melt_df, id_df], axis="columns")
            .assign(assay=assay, data_level=data_level, model=model)
        )
        
        # Replace the model variable with input data type for non-VAEs (it doesn't make sense otherwise)
        lsa_melt_df.loc[lsa_melt_df.input_data_type != "VAE", "model"] = (
            lsa_melt_df.loc[lsa_melt_df.input_data_type != "VAE", "input_data_type"]
        )
        
        lsa_dfs.append(lsa_melt_df)
        
lsa_dfs = pd.concat(lsa_dfs).reset_index(drop=True).dropna()

# Output file for downstream figure
output_file = pathlib.Path("data", "lsa_distribution_full_results.tsv.gz")
lsa_dfs.to_csv(output_file, sep="\t", index=False)

print(lsa_dfs.shape)
lsa_dfs.head()


# In[16]:


lsa_dfs.input_data_type.value_counts()


# In[17]:


lsa_dfs.input_data_type_full.value_counts()


# In[18]:


lsa_dfs.shuffled.value_counts()


# In[19]:


pd.crosstab(
    lsa_dfs.data_level,
    lsa_dfs.assay
)


# In[20]:


lsa_dfs.model.value_counts()

