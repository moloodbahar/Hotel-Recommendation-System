import pandas as pd
import numpy as np
%matplotlib inline
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

header=['uid','mid','rate']
ratings=pd.read_csv('C:/Users/marman/preprocessing/utilityMatrix_scale.csv',sep=',', header=None, skiprows=[0], names=header,error_bad_lines=False, engine='python')

def threshold_rates(df, uid_min, mid_min):
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
#     print('Ending rates info')
#     print('Number of users: {}'.format(n_users))
#     print('Number of models: {}'.format(n_items))
#     print('Sparsity: {:4.3f}%'.format(sparsity))
    return df
df_lim = threshold_rates(ratings, 5, 5)
# Create mappings
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

import pickle
import io
with open('featdlistU_newprofile.txt', 'rb') as handle:
    featdlistU = pickle.load(handle)

with open('feat_dlistI_newprofile.txt', 'rb') as handle:
    feat_dlistI = pickle.load(handle)

#vedtorize user_feature
from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer()
user_features = dv.fit_transform( featdlistU)
#user_features
# Need to hstack item_features
eye = sp.eye(user_features.shape[0], user_features.shape[0]).tocsr()
user_features_concat = sp.hstack((eye, user_features))
user_features_concat = user_features_concat.tocsr().astype(np.float32)
#user_features_concat


#vedtorize user_feature
from sklearn.feature_extraction import DictVectorizer
dvI = DictVectorizer()
item_features = dvI.fit_transform( feat_dlistI)
#item_features
# Need to hstack item_features
eye1 = sp.eye(item_features.shape[0], item_features.shape[0]).tocsr()
item_features_concat = sp.hstack((eye1, item_features))
item_features_concat = item_features_concat.tocsr().astype(np.float32)
#item_features_concat

def objective_wsideinfoUI(params):
    # unpack
    epochs, learning_rate,\
    no_components, item_alpha,\
    scale = params
    
    user_alpha = item_alpha * scale
    model = LightFM(loss='warp',
                    random_state=2016,
                    learning_rate=learning_rate,
                    no_components=no_components,
                    user_alpha=user_alpha,
                    item_alpha=item_alpha)
    model.fit(train, epochs=epochs,
              item_features=item_features_concat,user_features=user_features_concat,
              num_threads=6, verbose=True)
    
    #patks = lightfm.evaluation.precision_at_k(model, test,
       #                                       item_features=item_features_concat,
        #                                      train_interactions=None,
         #                                     k=5, num_threads=5)
    auc =lightfm.evaluation.auc_score(model,test,item_features=item_features_concat,user_features=user_features_concat,train_interactions=None,num_threads=6 )
    mapatk = np.mean(auc)
    # Make negative because we want to _minimize_ objective
    out = -mapatk
    # Weird shit going on
    if np.abs(out + 1) < 0.01 or out < -1.0:
        return 0.0
    else:
        return out

import pickle
filename = 'lightFM_model4_useritem_newProfile_both.sav'
model = pickle.load(open(filename, 'rb'))

user_representations = user_features_concat.dot(model.user_embeddings)
#user_representations

def predict_for_users(sims, norm=True):
    """Recommend products for all products"""
    pred = sims.dot(sims.T)
    if norm:
        norms = np.array([np.sqrt(np.diagonal(pred))])
        pred = pred / norms / norms.T
    return pred

user_similarities=predict_for_users(user_representations)
#user_similarities

def get_companynames(sim, name, idx_to_uid,uid_to_idx, N=10):
	"""
		This function return the name of the companies which have similar business policy to given company name.
		
		name = name of the input company
		sim =  matreix of similarity
		idx_to_uid = dictionary of mapped index to the name of the companies
		uid_to_idx = dictionary of mapped name of the companies to their indexes
		
	"""
    #name=name.upper()
    idx=uid_to_idx[name]
    row = sim[idx, :]
    companynames = []
    companysim={}
    count=0
    mostsimilarlist=np.argsort(-row)[:N]
    for x in np.argsort(-row)[:N]:
        companyname = idx_to_uid[x]
        companynames.append(companyname)
        idx_simmatrix=mostsimilarlist[count]
        companysim[companyname]=row[idx_simmatrix]
        count+=1
    return  companysim

user_representations = user_features_concat.dot(model.user_embeddings)
#user_representations

item_representations = item_features_concat.dot(model.item_embeddings)
#item_representations

def predict_for_customers(sim1,sim2):
    """Recommend items for all customers (user) in form of a matrix"""
    pred = sim1.dot(sim2.T)
    return pred

user2itemsim = predict_for_customers(user_representations,item_representations)
#user2itemsim

import math
def get_hotelnames2Users(sim, name,city_code,idx_to_mid,uid_to_idx,hnme_to_hid,N):
	""" This function return the top N recommended items to each specific customer in each specific city """
    count=0
    name=name.upper()
    idx=uid_to_idx[name]
    row = sim[idx, :]
    itemnames = []
    nameDict={}
    for x in np.argsort(-row):
        nameh=idx_to_mid[x]
        if nameh[2:5]==city_code:
            itemname = idx_to_mid[x]
            realname=hnme_to_hid[itemname]
            nameDict[itemname]=realname
            itemnames.append(nameDict)
            nameDict={}
            count+=1
            if count ==N+1:
                break 
    return itemnames

"""
	 craeting a mapping between the index and name of the hotels
	 to be able to choose the hotel in a specific city (post filtering)
	 
	 for each context for post filtering, something similar should be applied.
	 maybe there is a easier method, but I don't know! :P 
	 or really just using OLAP engine, it can be a fast solution
"""
	HotelDF = pd.read_csv('C:/Users/marman/preprocessing/HotelID_name.csv',sep=',', header=0, error_bad_lines=False, engine='python')
#print(HotelDF.head())
hid_to_hnme = {}
hnme_to_hid = {}
for index, row in HotelDF.iterrows():
    HOTEL_NAME=row['HOTEL_NAME']
    PROPERTY_ID=row['PROPERTY_ID']
    hid_to_hnme[HOTEL_NAME] = PROPERTY_ID
    hnme_to_hid[PROPERTY_ID] = HOTEL_NAME

def prompt(message, errormessage, isvalid):
    """Prompt for input given a message and return that value after verifying the input.

    Keyword arguments:
    message -- the message to display when asking the user for the value
    errormessage -- the message to display when the value fails validation
    isvalid -- a function that returns True if the value given by the user is valid
    """
    res = None
    while res is None:
        res = input(str(message)+': ')
        if not isvalid(res):
            print (errormessage)
            res = None
    return res
City_code = prompt(
        message = "Enter the name of the city code you want to use", 
        errormessage= "The city code should have three or at most four character length",
        isvalid = lambda v : len(v) == 3 or len(v) == 0 or len(v) == 4)
City_code = City_code.upper()
Comany_name =input("Enter the name of the company:")

Comany_name = Comany_name.upper()
get_hotelnames2Users(user2itemsim,Comany_name,City_code,idx_to_mid,uid_to_idx,hnme_to_hid, 10)

get_companynames(user_similarities,Comany_name,idx_to_uid,uid_to_idx, N=10)

