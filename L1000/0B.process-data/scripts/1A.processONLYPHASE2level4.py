#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import cmapPy.pandasGEXpress.parse as parse
import numpy as np
import matplotlib.pyplot as plt


# In[27]:


input_dir = '../0A.download-data/data/'
output_dir = 'data/'


# In[28]:


phase2_gctoo = parse.parse(input_dir + 'GSE70138_Broad_LINCS_Level4_ZSPCINF_mlr12k_n345976x12328_2017-03-06.gctx')


# In[29]:


phase2_data_df = phase2_gctoo.data_df.transpose()


# In[30]:


gene_info = pd.read_csv(input_dir + 'GSE92742_Broad_LINCS_gene_info.txt.gz', sep = "\t")
landmark_genes = np.char.mod('%d',gene_info[gene_info.pr_is_lm == 1].pr_gene_id)


# In[31]:


#slicing the dataframe to only include landmark genes
phase2_data_df = phase2_data_df.loc[:,phase2_data_df.columns.isin(landmark_genes)]


# In[32]:


#add pert_id and cell_id metadata column to phase2_data_df
phase2_sig_info = pd.read_csv(input_dir + 'GSE70138_Broad_LINCS_inst_info_2017-03-06.txt.gz', sep = "\t").set_index('inst_id').reindex(index=phase2_data_df.index)
phase2_sig_info = phase2_sig_info.loc[:,phase2_sig_info.columns.isin(['pert_id','cell_id'])]
phase2_df = pd.concat([phase2_sig_info,phase2_data_df], axis=1).reset_index()
phase2_df.head()


# In[33]:


# np.max(phase2_df['780'])


# In[34]:


# print(phase2_df.shape)
# phase2_df.groupby('cell_id').count().sort_values('cid')


# In[ ]:





# In[35]:


#add inchikey metadata column to phase2_data_df
# phase2_pert_info = pd.read_csv(input_dir + "GSE70138_Broad_LINCS_pert_info.txt.gz", sep = '\t').set_index('pert_id').reindex(index=phase2_df.pert_id)
# phase2_pert_info = phase2_pert_info.loc[:,phase2_pert_info.columns == 'inchi_key']
# phase2_pert_info['inchi_key'] = phase2_pert_info.inchi_key.str.split("-").str[0]
# phase2_pert_info = phase2_pert_info.rename(columns={"inchi_key": "inchi_key_prefix"})
# phase2_df = phase2_df.set_index("pert_id")
# phase2_df = pd.concat([phase2_pert_info, phase2_df], axis=1).reset_index()
# phase2_df.head()


# In[36]:


#combine phase1 and phase 2, and add rna_plate metadata feature
# combined_df = pd.concat([phase1_df,phase2_df])
# phase2_df['cid'] = phase2_df.cid.str.split(":").str[0]
# phase2_df = phase2_df.rename(columns={"cid": "rna_plate"})
# print(combined_df.shape)
# combined_df.head()


# In[37]:


# cell_painting_df = pd.read_csv('https://github.com/yuenler/cell-painting-vae/raw/master/0.preprocessing/data/cell_painting_complete.tsv.gz', sep = "\t")
# cell_painting_pert_id = cell_painting_df.Metadata_broad_sample.to_list()
# for i in range(len(cell_painting_pert_id)):
#     if cell_painting_pert_id[i][:3] == "BRD":
#         cell_painting_pert_id[i] = cell_painting_pert_id[i][:13]


# In[ ]:





# In[38]:


# # ids = pd.read_csv("data/repurposing_samples_20200324.txt", sep = "\t").loc[:,['broad_id','deprecated_broad_id']].dropna().reset_index(drop = True)
# cell_painting_inchi = pd.read_csv("https://github.com/broadinstitute/lincs-cell-painting/raw/master/metadata/moa/repurposing_info_external_moa_map_resolved.tsv", sep = "\t").loc[:,['broad_sample','InChIKey14']].dropna().set_index('broad_sample').reindex(index = cell_painting_df.Metadata_broad_sample)
# cell_painting_df = cell_painting_df.set_index("Metadata_broad_sample")
# cell_painting_df = pd.concat([cell_painting_inchi, cell_painting_df], axis=1).reset_index()
# # temp_ids = pd.DataFrame(columns = ['broad_id','deprecated_broad_id'])

# # for index, row in ids.iterrows():
# #     split_rows = row.deprecated_broad_id.split("|")
# #     for split_row in split_rows:
# #         temp_ids = temp_ids.append({'broad_id':row.broad_id, 'deprecated_broad_id':split_row}, ignore_index = True)

# # ids = pd.DataFrame(temp_ids)

# # ids.broad_id = ids.broad_id.apply(lambda x: str(x)[:13]).to_list()
# # ids.deprecated_broad_id = ids.deprecated_broad_id.apply(lambda x: str(x)[:13]).to_list()
# # ids = ids.set_index('deprecated_broad_id').to_dict()['broad_id']
# # len(ids)
# cell_painting_df


# In[39]:


# combined_df = combined_df.replace({"pert_id": ids})


# In[40]:


# for i in range(len(cell_painting_pert_id)):
#     if cell_painting_pert_id[i] in ids.keys():
#         cell_painting_pert_id[i] = ids[cell_painting_pert_id[i]]


# In[41]:


# cell_painting_inchi = cell_painting_df.InChIKey14.to_list()
# overlapped_df = combined_df[combined_df.inchi_key_prefix.isin(cell_painting_inchi)].reset_index(drop = True)
# print(overlapped_df.shape)
# overlapped_df.head()
# overlapped_df
# # overlapped_df = combined_df[combined_df.pert_id.isin(cell_painting_pert_id)].reset_index(drop = True)
# # print(overlapped_df.shape)
# # overlapped_df.head()
# # overlapped_df


# In[42]:


# A549_overlapped_df = overlapped_df[overlapped_df.cell_id == "A549"]
# print(A549_overlapped_df.shape)
# A549_overlapped_df.head()


# In[43]:


# hi = np.unique(overlapped_df.inchi_key_prefix.to_list())
# unique_cell= np.unique(cell_painting_inchi)
# count = 0
# nah = 0
# for i in range(len(unique_cell)):
#     if unique_cell[i] in hi:
#         count += 1
#     else:
#         nah += 1
# # cell_painting_pert_id
# print(count)
# print(nah)
# len(hi)
# len(np.unique(hi))


# In[44]:


# np.sort(cell_line_grouping.pert_id)


# In[45]:


# # overlap the right one when calculating
# highest_perts = cell_line_grouping.pert_id[cell_line_grouping.pert_id > 10000]
# highest_perts.plot.bar()


# In[46]:


# highest_inchi = cell_line_grouping.inchi_key_prefix[cell_line_grouping.inchi_key_prefix > 3500]
# highest_inchi.plot.bar()


# In[47]:


# cell_line_grouping_overlap = overlapped_df.groupby('cell_id').nunique()


# In[48]:


# cell_line_grouping_overlap.inchi_key_prefix


# In[49]:


# highest_perts_overlap = cell_line_grouping_overlap.pert_id[cell_line_grouping.pert_id > 10000]
# highest_perts_overlap.plot.bar()


# In[50]:


# highest_inchi_overlap = cell_line_grouping_overlap.inchi_key_prefix[cell_line_grouping_overlap.inchi_key_prefix > 700]
# highest_inchi_overlap.plot.bar()


# In[51]:


# A549_overlapped_df.to_csv(output_dir + 'L1000_A549_overlap.tsv.gz', sep = '\t', index = False)


# In[52]:


# overlapped_df.to_csv(output_dir + 'L1000_overlap.tsv.gz', sep = '\t', index = False)


# In[53]:


phase2_df.to_csv(output_dir + 'L1000_phase2.tsv.gz', sep = '\t', index = False)


# In[ ]:




