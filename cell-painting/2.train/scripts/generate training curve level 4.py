#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pathlib
import numpy as np
import pandas as pd
sys.path.insert(0, "../../scripts")
from utils import load_data


from pycytominer.cyto_utils import infer_cp_features


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.decomposition import PCA
from tensorflow import keras

from vae import VAE

from tensorflow.keras.models import Model, Sequential
import seaborn
import random as python_random
import tensorflow as tf
import umap


# In[ ]:





# In[2]:


data_splits = ["train", "test", "valid", "complete"]
data_dict = load_data(data_splits)


# In[3]:


# Prepare data for training
meta_features = infer_cp_features(data_dict["train"], metadata=True)
cp_features = infer_cp_features(data_dict["train"])

train_features_df = data_dict["train"].reindex(cp_features, axis="columns")
train_meta_df = data_dict["train"].reindex(meta_features, axis="columns")

test_features_df = data_dict["test"].reindex(cp_features, axis="columns")
test_meta_df = data_dict["test"].reindex(meta_features, axis="columns")

valid_features_df = data_dict["valid"].reindex(cp_features, axis="columns")
valid_meta_df = data_dict["valid"].reindex(meta_features, axis="columns")

complete_features_df = data_dict["complete"].reindex(cp_features, axis="columns")
complete_meta_df = data_dict["complete"].reindex(meta_features, axis="columns")


# In[4]:


print(train_features_df.shape)
train_features_df.head(3)


# In[5]:


def shuffle_each_column(df):
    columns = df.columns
    df_copy = df.copy()
    for column in columns:
        df_copy[column] = df_copy[column].sample(frac=1).reset_index(drop=True)
    return (df_copy)


# In[6]:


train_features_df = shuffle_each_column(train_features_df)
valid_features_df = shuffle_each_column(valid_features_df)


# In[7]:


encoder_architecture=[250]
decoder_architecture=[250]


# In[9]:


cp_vae = VAE(
    input_dim=train_features_df.shape[1],
    latent_dim=90,
    batch_size=32,
    encoder_batch_norm=True,
    epochs=50,
    learning_rate=0.0001,
    encoder_architecture=encoder_architecture,
    decoder_architecture=decoder_architecture,
    beta=0.06,
    verbose=True,
)
cp_vae.compile_vae()


# In[11]:


cp_vae.train(x_train=train_features_df, x_test=valid_features_df)


# In[13]:


cp_vae.vae


# In[15]:


# Save training performance
history_df = pd.DataFrame(cp_vae.vae.history.history)
history_df


# In[7]:


history_df.to_csv('level4_training_random.csv')


# In[8]:


history_df = pd.read_csv('level4_training_random.csv')


# In[9]:


original_training_data  = pd.read_csv('level4_training.csv')


# In[11]:


plt.figure(figsize=(7, 5), dpi = 400)
plt.plot(original_training_data["loss"], label="Training data")
plt.plot(original_training_data["val_loss"], label="Validation data")
plt.plot(history_df["loss"], label="Shuffled training data")
plt.plot(history_df["val_loss"], label="Shuffled validation data")
# plt.title("Loss for VAE training on Cell Painting Level 5 data")
plt.ylabel("MSE + KL Divergence")
plt.xlabel("No. Epoch")
plt.ylim(0,5)
plt.legend()
plt.show()


# In[8]:


cp_vae = VAE(
    input_dim=train_features_df.shape[1],
    latent_dim=90,
    batch_size=32,
    encoder_batch_norm=True,
    epochs=58,
    learning_rate=0.0001,
    encoder_architecture=encoder_architecture,
    decoder_architecture=decoder_architecture,
    beta=1,
    verbose=True,
)
cp_vae.compile_vae()


# In[9]:


cp_vae.train(x_train=train_features_df, x_test=valid_features_df)


# In[10]:


cp_vae.vae


# In[11]:


# Save training performance
history_df = pd.DataFrame(cp_vae.vae.history.history)
history_df


# In[12]:


history_df.to_csv('level4_training_vanilla_random.csv')


# In[12]:


# history_df = pd.read_csv('level4_training_vanilla_random.csv')


# In[13]:


original_training_data  = pd.read_csv('level4_training_vanilla.csv')


# In[14]:


plt.figure(figsize=(7, 5), dpi = 400)
plt.plot(original_training_data["loss"], label="Training data")
plt.plot(original_training_data["val_loss"], label="Validation data")
plt.plot(history_df["loss"], label="Shuffled training data")
plt.plot(history_df["val_loss"], label="Shuffled validation data")
# plt.title("Loss for VAE training on Cell Painting Level 5 data")
plt.ylabel("MSE + KL Divergence")
plt.xlabel("No. Epoch")
plt.ylim(0,5)
plt.legend()
plt.show()


# In[20]:


cp_vae = VAE(
    input_dim=train_features_df.shape[1],
    latent_dim=90,
    batch_size=32,
    encoder_batch_norm=True,
    epochs=50,
    learning_rate=0.0001,
    encoder_architecture=encoder_architecture,
    decoder_architecture=decoder_architecture,
    beta=0,
    lam = 10000,
    verbose=True,
)
cp_vae.compile_vae()


# In[21]:


cp_vae.train(x_train=train_features_df, x_test=valid_features_df)


# In[22]:


cp_vae.vae


# In[23]:


# Save training performance
history_df = pd.DataFrame(cp_vae.vae.history.history)
history_df


# In[24]:


history_df.to_csv('level4_training_mmd_random.csv')


# In[25]:


original_training_data  = pd.read_csv('level4_training_mmd.csv')


# In[26]:


plt.figure(figsize=(7, 5), dpi = 400)
plt.plot(original_training_data["loss"], label="Training data")
plt.plot(original_training_data["val_loss"], label="Validation data")
plt.plot(history_df["loss"], label="Shuffled training data")
plt.plot(history_df["val_loss"], label="Shuffled validation data")
# plt.title("Loss for VAE training on Cell Painting Level 5 data")
plt.ylabel("MSE + MMD")
plt.xlabel("No. Epoch")
plt.legend()
plt.show()


# In[ ]:




