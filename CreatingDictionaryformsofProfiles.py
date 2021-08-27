import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.special import expit
import pickle
import csv
import copy
import itertools
from lightfm import LightFM
import lightfm.evaluation
import sys

"""
	these cv files are the result of creating profiles for hotels and companies in previous step
"""
header=['uid','type','value']
sideinfoU = pd.read_csv('C:/Users/marman/preprocessing/Userprofile.csv',sep=',', header=None,skiprows=[0],names=header, error_bad_lines=False, engine='python')

header=['mid','type','value']
sideinfoI = pd.read_csv('C:/Users/marman/preprocessing/itemprofile.csv',sep=',', header=None,names=header, skiprows=[0], error_bad_lines=False, engine='python')
"""
	this csv file is the result of creating utility matrix and scaled value of them.
"""
header=['uid','mid','rate']
ratings=pd.read_csv('C:/Users/marman/preprocessing/utilityMatrix_scale.csv',sep=',', header=None, skiprows=[0], names=header,error_bad_lines=False, engine='python')


def threshold_rates(df, uid_min, mid_min):
	"""
		this function is considering a threshold for making the utility matrix more dense,
		I considered all of the companies with more than 5 times booking and considering hotels which they are booked at least 5 times.
	"""
    n_users = df.uid.unique().shape[0]
    n_items = df.mid.unique().shape[0]
    sparsity = float(df.shape[0]) / float(n_users*n_items) * 100
    print('Starting rates info')
    print('Number of users: {}'.format(n_users))
    print('Number of models: {}'.format(n_items))
    print('Sparsity: {:4.3f}%'.format(sparsity))
    
    done = False
    while not done:
        starting_shape = df.shape[0]
        mid_counts = df.groupby('uid').mid.count()
        df = df[~df.uid.isin(mid_counts[mid_counts < mid_min].index.tolist())]
        uid_counts = df.groupby('mid').uid.count()
        df = df[~df.mid.isin(uid_counts[uid_counts < uid_min].index.tolist())]
        ending_shape = df.shape[0]
        if starting_shape == ending_shape:
            done = True
    
    assert(df.groupby('uid').mid.count().min() >= mid_min)
    assert(df.groupby('mid').uid.count().min() >= uid_min)
    
    n_users = df.uid.unique().shape[0]
    n_items = df.mid.unique().shape[0]
    sparsity = float(df.shape[0]) / float(n_users*n_items) * 100
    print('Ending rates info')
    print('Number of users: {}'.format(n_users))
    print('Number of models: {}'.format(n_items))
    print('Sparsity: {:4.3f}%'.format(sparsity))
    return df
df_lim = threshold_rates(ratings, 5, 5)

"""
	this part is related to create an index mapping for names of companies and hotels.
"""
mid_to_idx = {}
idx_to_mid = {}
for (idx, mid) in enumerate(df_lim.mid.unique().tolist()):
    mid_to_idx[mid] = idx
    idx_to_mid[idx] = mid
    
uid_to_idx = {}
idx_to_uid = {}
for (idx, uid) in enumerate(df_lim.uid.unique().tolist()):
    uid_to_idx[uid] = idx
    idx_to_uid[idx] = uid

def map_ids(row, mapper):
    return mapper[row]

"""
	creating a sparse matrix in COOrdinate format which is the triplet format(user, item, value).
"""	
from scipy.sparse import coo_matrix
I = df_lim.uid.apply(map_ids, args=[uid_to_idx]).as_matrix()
J = df_lim.mid.apply(map_ids, args=[mid_to_idx]).as_matrix()
V = df_lim['rate'].values
likes = coo_matrix((V, (I, J)), dtype=np.float64)
likes = likes.tocsr()

import math
"""
	There's probably a fancy pandas groupby way to do
	this but I couldn't figure it out :(

	Build list of dictionaries containing features 
	and weights in same order as idx_to_mid prescribes.
	
	featdlistU_newprofile.txt : Saving the result of this transforming cpmpany profile to a list of dictionaries in a txt file.
"""
feat_dlistU = [{} for _ in idx_to_uid]
dictfeat={}
for idx, row in sideinfoU.iterrows():
    if row.type=='AvgPrice' :
        row.value = math.ceil(float(row.value)**(1/3))
        feat_key = '{}_{}'.format(row.type, str(row.value).lower())
    elif row.type=='Nbbooking':
        row.value = math.ceil(float(row.value)**(1/3))
        feat_key = '{}_{}'.format(row.type, str(row.value).lower())
    elif row.type=='Month':
        feat_key = '{}_{}'.format(row.type, str(row.value).lower())
    elif row.type=='Nb_city':
        feat_key = '{}_{}'.format(row.type, str(row.value).lower())
    elif row.type=='NoNight':
        row.value = math.ceil(float(row.value)**(1/2))
        feat_key = '{}_{}'.format(row.type, str(row.value).lower())
    else:
        feat_key = '{}'.format(row.type)
    idx =uid_to_idx.get(row.uid)
    if idx is not None:
            if '_' not in feat_key:
                feat_dlistU[idx][feat_key] = 1 #row.value in case of keeping different weights
            else:
                feat_dlistU[idx][feat_key] = 1
import pickle
with open('featdlistU_newprofile.txt', 'wb') as handle:
    pickle.dump(feat_dlistU, handle)
"""
	The same holds for items. 
	
	feat_dlistI_newprofile.txt : Saving the result of this transforming hotel profile to a list of dictionaries in a txt file.
"""	
feat_dlistI = [{} for _ in idx_to_mid]
for idx, row in sideinfoI.iterrows():
    if row.type=='AvgPrice' :
        row.value = math.ceil(float(row.value)**(1/3))
        feat_key = '{}_{}'.format(row.type, str(row.value).lower())
    elif row.type=='Nbbooking':
        row.value = math.ceil(float(row.value)**(1/3))
        feat_key = '{}_{}'.format(row.type, str(row.value).lower())
    elif row.type=='city':
        feat_key = '{}_{}'.format(row.type, str(row.value).lower())
    elif row.type=='AvgLeadTime':
        row.value = math.ceil(float(row.value))
        feat_key = '{}_{}'.format(row.type, str(row.value).lower())
    elif row.type=='NoNight':
        row.value = math.ceil(float(row.value)**(1/2))
        feat_key = '{}_{}'.format(row.type, str(row.value).lower())
    elif row.type=='AvgCommission':
        feat_key = '{}_{}'.format(row.type, str(row.value).lower())
    elif row.type=='suppliercode':
        feat_key = '{}_{}'.format(row.type, str(row.value).lower())
    elif row.type=='Month':
        feat_key = '{}_{}'.format(row.type, str(row.value).lower())
    else:
        feat_key = '{}'.format(row.type)
    idx =mid_to_idx.get(row.mid)
    if idx is not None:
            if '_' not in feat_key:
                feat_dlistI[idx][feat_key] = 1 #row.value in case of keeping different weights
            else:
                feat_dlistI[idx][feat_key] = 1
import pickle
with open('feat_dlistI_newprofile.txt', 'wb') as handle:
    pickle.dump(feat_dlistI, handle)
