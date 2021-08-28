#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from urllib.request import urlretrieve
import gzip
import shutil


# In[2]:


url = 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE70138&format=file&file=GSE70138%5FBroad%5FLINCS%5Finst%5Finfo%5F2017%2D03%2D06%2Etxt%2Egz'
name = 'GSE70138_Broad_LINCS_inst_info_2017-03-06.txt.gz'
path = os.path.join('data', name)


# In[3]:


urlretrieve(url, path)


# In[ ]:




