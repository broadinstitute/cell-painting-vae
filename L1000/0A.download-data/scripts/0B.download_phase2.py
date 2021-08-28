#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from urllib.request import urlretrieve
import gzip
import shutil


# In[2]:


url = 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE70138&format=file&file=GSE70138%5FBroad%5FLINCS%5FLevel5%5FCOMPZ%5Fn118050x12328%5F2017%2D03%2D06%2Egctx%2Egz'
name = 'GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx.gz'
path = os.path.join('data', name)


# In[3]:


urlretrieve(url, path)


# In[16]:


with gzip.open('data/GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx.gz', 'rb') as f_in:
    with open('data/GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

