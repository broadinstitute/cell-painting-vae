#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from urllib.request import urlretrieve
import gzip
import shutil


# In[2]:


url = 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE92742&format=file&file=GSE92742%5FBroad%5FLINCS%5FLevel5%5FCOMPZ%2EMODZ%5Fn473647x12328%2Egctx%2Egz'
name = 'GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz'
path = os.path.join('data', name)


# In[4]:


urlretrieve(url, path)


# In[5]:


with gzip.open('data/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz', 'rb') as f_in:
    with open('data/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

