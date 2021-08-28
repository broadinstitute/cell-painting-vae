#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import json

vis_datas = []
layers = ['onelayer','twolayer','threelayer']
for layer in layers:
    vis_data = []
    rootdir = 'parameter_sweepLEVEL5/' + layer
    for subdirs, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith("trial.json"):
                with open(subdirs + '/' + file, 'r') as json_file:
                    data = json_file.read()
                if 'loss' in json.loads(data)['metrics']['metrics']:
                    vis_data.append(json.loads(data))
    vis_datas.append(vis_data)


# In[6]:


optimal_hyperparameters = [vis_datas[0][0],vis_datas[1][0],vis_datas[2][0]]

for i in range(len(vis_datas)):
    for j in range(len(vis_datas[i])):
        if vis_datas[i][j]['metrics']['metrics']['val_loss']['observations'][0]['value'] < optimal_hyperparameters[i]['metrics']['metrics']['val_loss']['observations'][0]['value']:
            optimal_hyperparameters[i] = vis_datas[i][j]


# In[5]:


for layer in optimal_hyperparameters:
    print('latent_dim:', layer['hyperparameters']['values']['latent_dim'])
    print('learning_rate:', layer['hyperparameters']['values']['learning_rate'])
    print('beta:', layer['hyperparameters']['values']['beta'])
    print('encoder_batch_norm:', layer['hyperparameters']['values']['encoder_batch_norm'])
    print('batch_size:', layer['hyperparameters']['values']['batch_size'])
    print('epochs:', layer['hyperparameters']['values']['epochs'])
    print('loss:', layer['metrics']['metrics']['loss']['observations'][0]['value'])
    print('sval_loss:', layer['metrics']['metrics']['val_loss']['observations'][0]['value'])
    print()


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




