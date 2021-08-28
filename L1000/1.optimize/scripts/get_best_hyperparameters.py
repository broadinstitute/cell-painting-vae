#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import json
import hiplot as hip


# In[14]:


vis_datas = []
layers = ['twolayerL1000']
for layer in layers:
    vis_data = []
    rootdir = 'parameter_sweep/' + layer
    for subdirs, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith("trial.json"):
                with open(subdirs + '/' + file, 'r') as json_file:
                    data = json_file.read()
                if 'loss' in json.loads(data)['metrics']['metrics']:
                    vis_data.append(json.loads(data))
    vis_datas.append(vis_data)


# In[15]:


optimal_hyperparameters = [vis_datas[0][0]]
for i in range(len(vis_datas)):
    for j in range(len(vis_datas[i])):
        if vis_datas[i][j]['metrics']['metrics']['val_loss']['observations'][0]['value'] < optimal_hyperparameters[i]['metrics']['metrics']['val_loss']['observations'][0]['value']:
            optimal_hyperparameters[i] = vis_datas[i][j]


# In[16]:


for layer in optimal_hyperparameters:
    print('latent_dim:', layer['hyperparameters']['values']['latent_dim'])
    print('learning_rate:', layer['hyperparameters']['values']['learning_rate'])
#     print('encoder_batch_norm:', layer['hyperparameters']['values']['encoder_batch_norm'])
    print('batch_size:', layer['hyperparameters']['values']['batch_size'])
    print('epochs:', layer['hyperparameters']['values']['epochs'])
    print('loss:', layer['metrics']['metrics']['loss']['observations'][0]['value'])
    print('val_loss:', layer['metrics']['metrics']['val_loss']['observations'][0]['value'])
    print()


# In[18]:


vis_data = vis_data[1:]
data = [{'latent_dim': vis_data[idx]['hyperparameters']['values']['latent_dim'],
         'learning_rate': vis_data[idx]['hyperparameters']['values']['learning_rate'], 
#          'encoder_batch_norm': vis_data[idx]['hyperparameters']['values']['encoder_batch_norm'], 
         'batch_size': vis_data[idx]['hyperparameters']['values']['batch_size'],
         'epochs': vis_data[idx]['hyperparameters']['values']['epochs'], 
         'loss': vis_data[idx]['metrics']['metrics']['loss']['observations'][0]['value'],  
         'val_loss': vis_data[idx]['metrics']['metrics']['val_loss']['observations'][0]['value'], } for idx in range(len(vis_data))]

hip.Experiment.from_iterable(data).display()


# In[ ]:




