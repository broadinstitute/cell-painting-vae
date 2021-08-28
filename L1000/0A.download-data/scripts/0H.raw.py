#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
from urllib.request import urlretrieve
import zipfile
import shutil


# In[2]:


url = 'http://www.broadinstitute.org/bbbc/BBBC022/BBBC022_v1_images_20585w1.zip'
name = 'BBBC022_v1_images_20585w1.zip'
path = os.path.join('data', name)


# In[3]:


urlretrieve(url, path)


# In[5]:


with zipfile.ZipFile('data/BBBC022_v1_images_20585w1.zip', 'r') as zip_ref:
    zip_ref.extractall('data')

