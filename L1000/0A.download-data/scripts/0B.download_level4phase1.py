#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from urllib.request import urlretrieve
import gzip
import shutil


# In[3]:


url = 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE92742&format=file&file=GSE92742%5FBroad%5FLINCS%5FLevel4%5FZSPCINF%5Fmlr12k%5Fn1319138x12328%2Egctx%2Egz'
name = 'GSE92742_Broad_LINCS_Level4_ZSPCINF_mlr12k_n1319138x12328.gctx.gz'
path = os.path.join('data', name)


# In[ ]:


urlretrieve(url, path)


# In[ ]:


with gzip.open('data/GSE92742_Broad_LINCS_Level4_ZSPCINF_mlr12k_n1319138x12328.gctx.gz', 'rb') as f_in:
    with open('data/GSE92742_Broad_LINCS_Level4_ZSPCINF_mlr12k_n1319138x12328.gctx', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

