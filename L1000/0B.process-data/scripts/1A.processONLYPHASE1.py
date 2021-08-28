#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import cmapPy.pandasGEXpress.parse as parse
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


input_dir = '../0A.download-data/data/'
output_dir = 'data/'


# In[3]:


phase1_gctoo = parse.parse(input_dir + 'GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx')


# In[4]:


phase1_data_df = phase1_gctoo.data_df.transpose()


# In[5]:


gene_info = pd.read_csv(input_dir + 'GSE92742_Broad_LINCS_gene_info.txt.gz', sep = "\t")
landmark_genes = np.char.mod('%d',gene_info[gene_info.pr_is_lm == 1].pr_gene_id)


# In[6]:


#slicing the dataframe to only include landmark genes
phase1_data_df = phase1_data_df.loc[:,phase1_data_df.columns.isin(landmark_genes)]


# In[7]:


#add pert_id and cell_id metadata column to phase1_data_df
phase1_sig_info = pd.read_csv(input_dir + 'GSE92742_Broad_LINCS_sig_info.txt.gz', sep = "\t").set_index('sig_id').reindex(index=phase1_data_df.index)
phase1_sig_info = phase1_sig_info.loc[:,phase1_sig_info.columns.isin(['pert_id','cell_id'])]
phase1_df = pd.concat([phase1_sig_info,phase1_data_df], axis=1).reset_index()
phase1_df.head()


# In[8]:


#add inchikey metadata column to phase1_data_df
phase1_pert_info = pd.read_csv(input_dir + "GSE92742_Broad_LINCS_pert_info.txt.gz", sep = '\t').set_index('pert_id').reindex(index=phase1_df.pert_id)
phase1_pert_info = phase1_pert_info.loc[:,phase1_pert_info.columns == 'inchi_key_prefix']
phase1_df = phase1_df.set_index("pert_id")
phase1_df = pd.concat([phase1_pert_info, phase1_df], axis=1).reset_index()
phase1_df.head()


# In[9]:


phase1_df.to_csv(output_dir + 'L1000_phase1.tsv.gz', sep = '\t', index = False)

