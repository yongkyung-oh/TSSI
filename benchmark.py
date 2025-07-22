#-*- coding:utf-8 -*-

import os
import sys
import time
import datetime
import pickle
import random

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms

# Set random seed
SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# sktime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from sktime.datasets import load_UCR_UEA_dataset
# from sktime.transformers.series_as_features.compose import RowTransformer
from sktime_dl.classification import *
from lib import load_from_tsfile_to_dataframe

# get different data
data_info = pd.read_csv('dataset_summary_multivariate.csv', index_col=0)
dataname_list = data_info['problem'].tolist()
dataname_list = np.unique(dataname_list)

if not os.path.exists('out'):
    os.mkdir('out')
    
for dataname in dataname_list:
    if not os.path.exists(os.path.join('out', dataname)):
        os.mkdir(os.path.join('out', dataname))

# skip if output exists
dataname_selected = []
for dataname in dataname_list:
    for k in range(10):
        for model in ['CNN', 'FCN', 'Inception', 'LSTMFCN', 'Encoder', 'MCDCNN', 'MLP', 'ResNet', 'TLENET', 'TapNet', 'MACNN']:
            if os.path.exists(os.path.join('out', dataname, '{}_{}'.format(model, k))):
                continue
            else:
                dataname_selected.append(dataname)

dataname_selected = np.unique(dataname_selected)
print(dataname_selected)

### 
for dataname in dataname_selected:
    try:
        try:
            X_train, y_train = load_UCR_UEA_dataset(dataname, 'train', return_X_y=True)
            X_test, y_test = load_UCR_UEA_dataset(dataname, 'test', return_X_y=True)
        except:
            X_train, y_train = load_from_tsfile_to_dataframe(os.path.join('data/Multivariate_ts', "{}/{}_TRAIN.ts".format(dataname,dataname)))
            X_test, y_test = load_from_tsfile_to_dataframe(os.path.join('data/Multivariate_ts', "{}/{}_TEST.ts".format(dataname,dataname)))

        label_dict = dict(zip(np.unique(y_train), range(len(np.unique(y_train)))))
        y_train = np.array([label_dict[y] for y in y_train])
        y_test = np.array([label_dict[y] for y in y_test])

        num_dim = X_train.shape[1]
        num_class = len(np.unique(y_train))
    except:
        print(dataname)
        continue
        
    # interpolate to resize
    try:
        len_list = []
        for i in range(X_train.shape[0]):
            for j in range(X_train.shape[1]):
                len_list.append(len(X_train.iloc[i,j]))

        def get_same_len(s):
            s_re = np.interp(
                   np.arange(0,max_len),
                   np.linspace(0,max_len,num=len(s)),
                   s)
            return pd.Series(s_re)
                
        if min(len_list) != max(len_list):
            max_len = max(len_list)
            X_train = X_train.applymap(lambda s: get_same_len(s))
            X_test = X_test.applymap(lambda s: get_same_len(s))
        else:
            pass   
    except:
        print(dataname)
        continue
        
    print(dataname, len(X_train), len(X_test), num_dim, num_class, max(len_list))
                

    ## z-normalized
    X_train_np = []
    for i in range(X_train.shape[1]):
        X_train_np.append(pd.concat(X_train.iloc[:,i].tolist(), axis=1).fillna(0).T.to_numpy())
    X_train_np = np.stack(X_train_np, axis=1) # [N,D,S]

    stat = X_train_np.transpose(1,0,2).reshape(num_dim,-1) # [D,N,S] -> [D]
    mean = stat.mean(axis=-1)
    std = stat.std(axis=-1)

    X_train_norm = []
    for i in range(X_train.shape[1]):
        X_train_norm.append(X_train.iloc[:,[i]].applymap(lambda s: (s-mean[i]) / (std[i] + 1e-5)).fillna(0))
    X_train_norm = pd.concat(X_train_norm, axis=1)

    X_test_norm = []
    for i in range(X_test.shape[1]):
        X_test_norm.append(X_test.iloc[:,[i]].applymap(lambda s: (s-mean[i]) / (std[i] + 1e-5)).fillna(0))
    X_test_norm = pd.concat(X_test_norm, axis=1)

    X_train = X_train_norm
    X_test = X_test_norm
    
    ## run model 
    for k in range(10):
        for model in ['CNN', 'FCN', 'Inception', 'LSTMFCN', 'Encoder', 'MCDCNN', 'MLP', 'ResNet', 'TLENET', 'TapNet', 'MACNN']:
            # check exist
            if os.path.exists(os.path.join('out', dataname, '{}_{}'.format(model, k))):
                continue
            else:
                pass

            # run model
            nb_epoch = 100
            batch_size = 16 if len(X_train) > 16 else 4

            if model == 'CNN': 
                network = CNNClassifier(nb_epochs=nb_epoch, batch_size=batch_size, verbose=False)
            elif model == 'FCN':
                network = FCNClassifier(nb_epochs=nb_epoch, batch_size=batch_size, verbose=False)
            elif model == 'Inception':
                network = InceptionTimeClassifier(nb_epochs=nb_epoch, batch_size=batch_size, verbose=False)
            elif model == 'LSTMFCN':
                network = LSTMFCNClassifier(nb_epochs=nb_epoch, batch_size=batch_size, verbose=False)
            elif model == 'Encoder':
                network = EncoderClassifier(nb_epochs=nb_epoch, batch_size=batch_size, verbose=False)
            elif model == 'MCDCNN':
                network = MCDCNNClassifier(nb_epochs=nb_epoch, batch_size=batch_size, verbose=False)
            elif model == 'MLP':
                network = MLPClassifier(nb_epochs=nb_epoch, batch_size=batch_size, verbose=False)
            elif model == 'ResNet':
                network = ResNetClassifier(nb_epochs=nb_epoch, batch_size=batch_size, verbose=False)
            elif model == 'TLENET':
                network = TLENETClassifier(nb_epochs=nb_epoch, batch_size=batch_size, verbose=False)
            elif model == 'TapNet':
                network = TapNetClassifier(nb_epochs=nb_epoch, batch_size=batch_size, verbose=False)
            elif model == 'MACNN':
                network = MACNNClassifier(nb_epochs=nb_epoch, batch_size=batch_size, verbose=False)

            try:
                network.fit(X_train, y_train)
                y_pred = network.predict(X_test)
                y_proba = network.predict_proba(X_test)

                with open(os.path.join('out', dataname, '{}_{}'.format(model, k)), 'wb') as f:
                    pickle.dump([y_test, y_pred, y_proba], f)

                print(dataname, model, network.score(X_test, y_test), f1_score(y_test, y_pred, average='weighted'))

            except:
                continue
