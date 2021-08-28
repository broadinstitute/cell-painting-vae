#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from urllib.request import urlretrieve
import gzip
import shutil


# In[3]:


url = 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE70138&format=file&file=GSE70138%5FBroad%5FLINCS%5Fpert%5Finfo%2Etxt%2Egz'
name = 'GSE70138_Broad_LINCS_pert_info.txt.gz'
path = os.path.join('data', name)


# In[4]:


urlretrieve(url, path)

