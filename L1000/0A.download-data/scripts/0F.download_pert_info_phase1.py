#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from urllib.request import urlretrieve
import gzip
import shutil


# In[2]:


url = 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE92742&format=file&file=GSE92742%5FBroad%5FLINCS%5Fpert%5Finfo%2Etxt%2Egz'
name = 'GSE92742_Broad_LINCS_pert_info.txt.gz'
path = os.path.join('data', name)


# In[3]:


urlretrieve(url, path)

