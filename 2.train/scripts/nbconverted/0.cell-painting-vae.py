#!/usr/bin/env python
# coding: utf-8

# # Train a VAE on Cell Painting LINCS Data

# In[53]:


import sys
import pathlib
import numpy as np
import pandas as pd

from tensorflow import keras

from pycytominer.cyto_utils import infer_cp_features

sys.path.insert(0, "../scripts")
from utils import load_data
from vae import VAE
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, Sequential
import seaborn
import random as python_random
import tensorflow as tf


# In[54]:


get_ipython().run_line_magic('load_ext', 'nb_black')


# In[55]:


# np.random.seed(123)
# python_random.seed(123)
# tf.random.set_random_seed(1234)


# In[56]:


data_splits = ["train", "test", "complete"]
data_dict = load_data(data_splits)


# In[57]:


# Prepare data for training
meta_features = infer_cp_features(data_dict["train"], metadata=True)
cp_features = infer_cp_features(data_dict["train"])

train_features_df = data_dict["train"].reindex(cp_features, axis="columns")
train_meta_df = data_dict["train"].reindex(meta_features, axis="columns")

test_features_df = data_dict["test"].reindex(cp_features, axis="columns")
test_meta_df = data_dict["test"].reindex(meta_features, axis="columns")

complete_features_df = data_dict["complete"].reindex(cp_features, axis="columns")
complete_meta_df = data_dict["complete"].reindex(meta_features, axis="columns")


# In[58]:


print(train_features_df.shape)
train_features_df.head(3)


# In[59]:


print(test_features_df.shape)
test_features_df.head(3)


# In[60]:


print(complete_features_df.shape)
complete_features_df.head(3)


# In[61]:


# VAE of one layer
encoder_architecture = []
decoder_architecture = []

# VAE of two layers
# encoder_architecture = [100]
# decoder_architecture = [100]


# In[62]:


# model from optimal hyperparameters for onelayer obtained from 1.optimize
cp_vae = VAE(
    input_dim=train_features_df.shape[1],
    latent_dim=5,
    batch_size=32,
    encoder_batch_norm=True,
    epochs=14,
    learning_rate=0.01,
    encoder_architecture=encoder_architecture,
    decoder_architecture=decoder_architecture,
    beta=1,
    verbose=True,
)
cp_vae.compile_vae()

# model from optimal hyperparameters for twolayer obtained from 1.optimize
# cp_vae = VAE(
#     input_dim=train_features_df.shape[1],
#     latent_dim=30,
#     batch_size=128,
#     encoder_batch_norm=True,
#     epochs=14,
#     learning_rate=0.01,
#     encoder_architecture=encoder_architecture,
#     decoder_architecture=decoder_architecture,
#     beta=1.0,
#     verbose=True,
# )
# cp_vae.compile_vae()


# In[63]:


cp_vae.train(x_train=train_features_df, x_test=test_features_df)


# In[64]:


cp_vae.vae


# In[65]:


# Save training performance
history_df = pd.DataFrame(cp_vae.vae.history.history)
history_df


# In[66]:


plt.figure(figsize=(10, 5))
plt.plot(history_df["loss"], label="Training data")
plt.plot(history_df["val_loss"], label="Validation data")
plt.title("Loss for VAE training on cell painting data")
plt.ylabel("Binary cross entropy + KL Divergence")
plt.xlabel("No. Epoch")
plt.legend()
plt.show()


# In[67]:


#latent space heatmap
fig, ax = plt.subplots(figsize=(10, 10))
encoder = cp_vae.encoder_block["encoder"]
latent = np.array(encoder.predict(test_features_df)[2])
seaborn.heatmap(latent, ax=ax)


# In[68]:


#original vs reconstructed heatmap
reconstruction = pd.DataFrame(cp_vae.vae.predict(test_features_df), columns=cp_features)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
seaborn.heatmap(test_features_df, ax=ax1)
seaborn.heatmap(reconstruction, ax=ax2)
plt.show()


# In[69]:


#difference between original and reconstructed heatmap
difference = abs(reconstruction - test_features_df)
fig, ax = plt.subplots(figsize=(10, 10))
seaborn.heatmap(difference, ax=ax, cmap="Blues")


# In[70]:


#encoder heatmap
weights = cp_vae.encoder_block["encoder"].get_weights()
fig, ax = plt.subplots(figsize=(10, 10))
seaborn.heatmap(weights[0], ax=ax)


# In[71]:


#NOTE: IF YOU RUN THIS, YOU WILL NOT BE ABLE TO REPRODUCE THE EXACT RESULTS IN THE EXPERIMENT
# latent_complete = np.array(encoder.predict(complete_features_df)[2])
# latent_df = pd.DataFrame(latent_complete)
# latent_df.to_csv("../3.application/latent.csv")


# In[72]:


#NOTE: IF YOU RUN THIS, YOU WILL NOT BE ABLE TO REPRODUCE THE EXACT RESULTS IN THE EXPERIMENT
# decoder = cp_vae.decoder_block["decoder"]
# decoder.save("decoder")

