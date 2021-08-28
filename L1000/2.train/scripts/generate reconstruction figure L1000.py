#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pathlib
import numpy as np
import pandas as pd
sys.path.insert(0, "../../scripts")
from utils import load_data, infer_L1000_features
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.decomposition import PCA

from tensorflow import keras


from vae import VAE
from tensorflow.keras.models import Model, Sequential
import seaborn
import tensorflow as tf
import umap
import seaborn as sns


# In[2]:


data_splits = ["train", "test", "valid", "complete"]
data_dict = load_data(data_splits, dataset="L1000")


# In[3]:


# Prepare data for training

meta_features = infer_L1000_features(data_dict["train"], metadata=True)
profile_features = infer_L1000_features(data_dict["train"])

train_features_df = data_dict["train"].reindex(profile_features, axis="columns")
train_meta_df = data_dict["train"].reindex(meta_features, axis="columns")

valid_features_df = data_dict["valid"].reindex(profile_features, axis="columns")
valid_meta_df = data_dict["valid"].reindex(meta_features, axis="columns")

test_features_df = data_dict["test"].reindex(profile_features, axis="columns")
test_meta_df = data_dict["test"].reindex(meta_features, axis="columns")

complete_features_df = data_dict["complete"].reindex(profile_features, axis="columns")
complete_meta_df = data_dict["complete"].reindex(meta_features, axis="columns")


# In[4]:


decoder_beta = keras.models.load_model("models/L1000twolayerDecoder")
encoder_beta = keras.models.load_model("models/L1000twolayerEncoder")
decoder_vanilla = keras.models.load_model("models/L1000twolayerDecoder_vanilla")
encoder_vanilla = keras.models.load_model("models/L1000twolayerEncoder_vanilla")
decoder_mmd = keras.models.load_model("models/L1000twolayerDecoder_mmd")
encoder_mmd = keras.models.load_model("models/L1000twolayerEncoder_mmd")


# In[5]:


reconstruction_beta = pd.DataFrame(decoder_beta.predict(encoder_beta.predict(test_features_df)[2]))
reconstruction_beta['label'] = 'β-VAE reconstruction'
reconstruction_vanilla = pd.DataFrame(decoder_vanilla.predict(encoder_vanilla.predict(test_features_df)[2]))
reconstruction_vanilla['label'] = 'Vanilla VAE reconstruction'
reconstruction_mmd = pd.DataFrame(decoder_mmd.predict(encoder_mmd.predict(test_features_df)[2]))
reconstruction_mmd['label'] = 'MMD-VAE reconstruction'


# In[6]:


simulated_test_df = pd.DataFrame(np.random.normal(size=(11805, 65)), columns=np.arange(0,65))
reconstruction_of_simulated_test_beta = pd.DataFrame(decoder_beta.predict(simulated_test_df))
reconstruction_of_simulated_test_beta['label'] = 'β-VAE simulation'
reconstruction_of_simulated_test_vanilla = pd.DataFrame(decoder_vanilla.predict(simulated_test_df))
reconstruction_of_simulated_test_vanilla['label'] = 'Vanilla VAE simulation'

reconstruction_of_simulated_test_mmd = pd.DataFrame(decoder_mmd.predict(simulated_test_df))
reconstruction_of_simulated_test_mmd['label'] = 'MMD-VAE simulation'
test_features_df.columns = np.arange(0,978)
test_features_df['label'] = 'Original'


# In[7]:


beta_df = pd.concat([test_features_df, reconstruction_beta,reconstruction_of_simulated_test_beta])
mmd_df = pd.concat([test_features_df,reconstruction_mmd,reconstruction_of_simulated_test_mmd])
vanilla_df = pd.concat([test_features_df,reconstruction_vanilla,reconstruction_of_simulated_test_vanilla])

labels_beta = beta_df.label.reset_index(drop=True)
labels_mmd = mmd_df.label.reset_index(drop=True)
labels_vanilla = vanilla_df.label.reset_index(drop=True)

beta_df = beta_df.drop('label', axis = 1)
mmd_df = mmd_df.drop('label', axis = 1)
vanilla_df = vanilla_df.drop('label', axis = 1)


# In[76]:


reducer = umap.UMAP(random_state=123, min_dist = 0.5, n_neighbors=5).fit(test_features_df.drop('label', axis=1))


# In[77]:


original_embedding = pd.DataFrame(reducer.transform(test_features_df.drop('label', axis=1)))


# In[78]:


sns.scatterplot(data=original_embedding, x = 0, y=1, alpha=0.5, s=10)


# In[60]:


embedding_beta = pd.DataFrame(reducer.transform(beta_df))
embedding_mmd = pd.DataFrame(reducer.transform(mmd_df))
embedding_vanilla = pd.DataFrame(reducer.transform(vanilla_df))


# In[61]:


embedding_beta = pd.concat([embedding_beta, labels_beta], axis =1 ).rename(columns = {'label': ''})
embedding_mmd = pd.concat([embedding_mmd, labels_mmd], axis =1 ).rename(columns = {'label': ''})
embedding_vanilla = pd.concat([embedding_vanilla, labels_vanilla], axis =1 ).rename(columns = {'label': ''})


# In[62]:


sns.set(font_scale=1)
sns.set_style("darkgrid")


# In[72]:


fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(18, 6), dpi=400)
sns.scatterplot(data=embedding_vanilla, ax=ax1, x = 0, y=1, hue='', alpha=0.5, s=1
)
ax1.set_xlabel('UMAP 1')
ax1.set_ylabel('UMAP 2')

sns.scatterplot(data=embedding_beta, ax=ax2, x = 0, y=1, hue='', alpha=0.5, s=1
)
ax2.set_xlabel('UMAP 1')
ax2.set_ylabel('UMAP 2')

sns.scatterplot(data=embedding_mmd, ax=ax3, x = 0, y=1, hue='', alpha=0.5, s=1
)
ax3.set_xlabel('UMAP 1')
ax3.set_ylabel('UMAP 2')

plt.show()


# In[ ]:




