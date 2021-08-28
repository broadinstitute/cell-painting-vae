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


encoder_architecture = [500]
decoder_architecture = [500]


# In[8]:


L1000_vae = VAE(
    input_dim=train_features_df.shape[1],
    latent_dim=65,
    batch_size=512,
    encoder_batch_norm=True,
    epochs=180,
    learning_rate=0.001,
    encoder_architecture=encoder_architecture,
    decoder_architecture=decoder_architecture,
    beta=40,
    verbose=True,
)

L1000_vae.compile_vae()


# In[9]:


L1000_vae.train(x_train=train_features_df, x_test=valid_features_df)


# In[10]:


L1000_vae.vae


# In[11]:


# Save training performance
history_df = pd.DataFrame(L1000_vae.vae.history.history)
history_df


# In[12]:


history_df.to_csv('L1000_training_random.csv')


# In[14]:


original_training_data  = pd.read_csv('twolayer_training.csv')


# In[15]:


plt.figure(figsize=(7, 5), dpi = 400)
plt.plot(original_training_data["loss"], label="Training data")
plt.plot(original_training_data["val_loss"], label="Validation data")
plt.plot(history_df["loss"], label="Shuffled training data")
plt.plot(history_df["val_loss"], label="Shuffled validation data")
# plt.title("Loss for VAE training on Cell Painting Level 5 data")
plt.ylabel("MSE + KL Divergence")
plt.xlabel("No. Epoch")
plt.legend()
plt.show()


# In[16]:


L1000_vae = VAE(
    input_dim=train_features_df.shape[1],
    latent_dim=65,
    batch_size=512,
    encoder_batch_norm=True,
    epochs=180,
    learning_rate=0.001,
    encoder_architecture=encoder_architecture,
    decoder_architecture=decoder_architecture,
    beta=1,
    verbose=True,
)

L1000_vae.compile_vae()


# In[18]:


L1000_vae.train(x_train=train_features_df, x_test=valid_features_df)


# In[20]:


L1000_vae.vae


# In[22]:


# Save training performance
history_df = pd.DataFrame(L1000_vae.vae.history.history)
history_df


# In[23]:


history_df.to_csv('L1000_training_vanilla_random.csv')


# In[24]:


original_training_data  = pd.read_csv('twolayer_training_vanilla.csv')


# In[25]:


plt.figure(figsize=(7, 5), dpi = 400)
plt.plot(original_training_data["loss"], label="Training data")
plt.plot(original_training_data["val_loss"], label="Validation data")
plt.plot(history_df["loss"], label="Shuffled training data")
plt.plot(history_df["val_loss"], label="Shuffled validation data")
# plt.title("Loss for VAE training on Cell Painting Level 5 data")
plt.ylabel("MSE + KL Divergence")
plt.xlabel("No. Epoch")
plt.legend()
plt.show()


# In[8]:


L1000_vae = VAE(
    input_dim=train_features_df.shape[1],
    latent_dim=65,
    batch_size=512,
    encoder_batch_norm=True,
    epochs=30,
    learning_rate=0.001,
    encoder_architecture=encoder_architecture,
    decoder_architecture=decoder_architecture,
    beta=0,
    lam=10000000,
    verbose=True,
)

L1000_vae.compile_vae()


# In[9]:


L1000_vae.train(x_train=train_features_df, x_test=valid_features_df)


# In[10]:


L1000_vae.vae


# In[11]:


# Save training performance
history_df = pd.DataFrame(L1000_vae.vae.history.history)
history_df


# In[12]:


history_df.to_csv('L1000_training_mmd_random.csv')


# In[14]:


original_training_data  = pd.read_csv('twolayer_training_mmd.csv')


# In[15]:


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




