#!/usr/bin/env python
# coding: utf-8

# In[2]:


import kerastuner as kt
import tensorflow as tf
import subprocess
import optimize
import sys
sys.path.insert(0, "../scripts")
from utils import load_data
from kerastuner.tuners import BayesianOptimization
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from pycytominer.cyto_utils import infer_cp_features
from optimize_utils import HyperVAE, CustomBayesianTuner, get_optimize_args
from vae import VAE


# In[3]:


architectures = ["onelayer","twolayer","threelayer"]
tuners = []
# Load Data
data_splits = ["train", "test"]
data_dict = load_data(data_splits)

# Prepare data for training
meta_features = infer_cp_features(data_dict["train"], metadata=True)
cp_features = infer_cp_features(data_dict["train"])

train_features_df = data_dict["train"].reindex(cp_features, axis="columns")
train_meta_df = data_dict["train"].reindex(meta_features, axis="columns")

test_features_df = data_dict["test"].reindex(cp_features, axis="columns")
test_meta_df = data_dict["test"].reindex(meta_features, axis="columns")

for i in range(len(architectures)):
    if architectures[i] == "onelayer":
        encoder_architecture = []
        decoder_architecture = []
    if architectures[i] == "twolayer":
        encoder_architecture = [100]
        decoder_architecture = [100]
    if architectures[i] == "threelayer":
        encoder_architecture = [250, 100]
        decoder_architecture = [100, 250]

    # Initialize hyper parameter VAE tuning
    hypermodel = HyperVAE(
        input_dim=train_features_df.shape[1],
        min_latent_dim=optimize.min_latent_dim,
        max_latent_dim=optimize.max_latent_dim,
        min_beta=optimize.min_beta,
        max_beta=optimize.max_beta,
        learning_rate=optimize.learning_rate,
        encoder_batch_norm=True,
        encoder_architecture=encoder_architecture,
        decoder_architecture=decoder_architecture,
    )

    tuners.append(CustomBayesianTuner(
        hypermodel,
        objective="val_loss",
        max_trials=1000,
        directory="parameter_sweep",
        project_name=architectures[i],
        overwrite=False,
    ))


# In[4]:


train_loss = []
test_loss = []
best_hyperparameters = []
for i in range(len(tuners)):
    best_model = tuners[i].get_best_models(num_models=1)[0]
    train_loss.append(best_model.evaluate(train_features_df))
    test_loss.append(best_model.evaluate(test_features_df))
    
    best_hyperparameters.append(tuners[i].get_best_hyperparameters(num_trials = 1)[0])

for i in range(len(tuners)):
    print("\n\n")
    print("best model train loss for", architectures[i] + ":", train_loss[i])
    print("best model test loss for", architectures[i] + ":", test_loss[i])
    print()
    print("best hyperparameters for", architectures[i] + ":")
    hyperparameter_names = ['batch_size', 'beta', 'encoder_batch_norm', 'epochs', 'latent_dim', 'learning_rate']
    for hyperparameter_name in hyperparameter_names:
        print(hyperparameter_name + ":", best_hyperparameters[i].get(hyperparameter_name))


# In[ ]:





# In[5]:


import os
import json

vis_data = []
rootdir = 'parameter_sweep/onelayer'
for subdirs, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith("trial.json"):
            with open(subdirs + '/' + file, 'r') as json_file:
                data = json_file.read()
            vis_data.append(json.loads(data))


# In[6]:


import hiplot as hip
data = [{'latent_dim': vis_data[idx]['hyperparameters']['values']['latent_dim'],
         'learning_rate': vis_data[idx]['hyperparameters']['values']['learning_rate'], 
         'beta': vis_data[idx]['hyperparameters']['values']['beta'], 
         'encoder_batch_norm': vis_data[idx]['hyperparameters']['values']['encoder_batch_norm'], 
         'batch_size': vis_data[idx]['hyperparameters']['values']['batch_size'],
         'epochs': vis_data[idx]['hyperparameters']['values']['epochs'], 
         'loss': vis_data[idx]['metrics']['metrics']['loss']['observations'][0]['value'],  
         'val_loss': vis_data[idx]['metrics']['metrics']['val_loss']['observations'][0]['value'], } for idx in range(len(vis_data))]

hip.Experiment.from_iterable(data).display()


# In[ ]:




