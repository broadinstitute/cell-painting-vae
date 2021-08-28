#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


latent_level5_beta = pd.read_csv('level5Latent.csv').drop('Unnamed: 0', axis =1)
latent_level5_vanilla = pd.read_csv('level5Latent_vanilla.csv').drop('Unnamed: 0', axis =1)
latent_level5_mmd = pd.read_csv('level5Latent_mmd.csv').drop('Unnamed: 0', axis =1)


# In[3]:


correlation_beta = latent_level5_beta.corr().abs()
correlation_vanilla = latent_level5_vanilla.corr().abs()
correlation_mmd = latent_level5_mmd.corr().abs()


# In[7]:


fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(18, 5), dpi=500, sharey = True)

sns.heatmap(correlation_vanilla, ax=ax1, cmap="BuPu")
ax1.set_xlabel('Vanilla VAE')
sns.heatmap(correlation_beta, ax=ax2, cmap="BuPu")
ax2.set_xlabel('β-VAE')
sns.heatmap(correlation_mmd, ax=ax3, cmap="BuPu")
ax3.set_xlabel('MMD-VAE')

