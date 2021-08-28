#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
from urllib.request import urlretrieve
import gzip
import shutil


# In[10]:


url = 'https://ndownloader.figshare.com/articles/13181966/versions/1'
# name = 'adenyiData.gctx'
path = os.path.join('data')


# In[11]:


urlretrieve(url, path)


# In[ ]:


# with gzip.open('data/adenyiData.gctx.gz', 'rb') as f_in:
#     with open('data/adenyiData.gctx', 'wb') as f_out:
#         shutil.copyfileobj(f_in, f_out)

