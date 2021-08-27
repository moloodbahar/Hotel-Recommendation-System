import pandas as pd
import numpy as np
#%matplotlib inline
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
    print('Ending rates info')
    print('Number of users: {}'.format(n_users))
    print('Number of models: {}'.format(n_items))
    print('Sparsity: {:4.3f}%'.format(sparsity))
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

def map_ids(row, mapper):
    return mapper[row]

from scipy.sparse import coo_matrix
I = df_lim.uid.apply(map_ids, args=[uid_to_idx]).as_matrix()
J = df_lim.mid.apply(map_ids, args=[mid_to_idx]).as_matrix()
V = df_lim['rate'].values
likes = coo_matrix((V, (I, J)), dtype=np.float64)
likes = likes.tocsr()


from scipy.sparse import lil_matrix
def train_test_split(ratings, split_count, fraction=None):
    """
    Split recommendation data into train and test sets
    
    Params
    ------
    ratings : scipy.sparse matrix
        Interactions between users and items.
    split_count : int
        Number of user-item-interactions per user to move
        from training to test set.
    fractions : float
        Fraction of users to split off some of their
        interactions into test set. If None, then all 
        users are considered.
    """
    # Note: likely not the fastest way to do things below.
    train = ratings.copy().tocoo()
    test = lil_matrix(train.shape)
    
    if fraction:
        try:
            user_index = np.random.choice(
                np.where(np.bincount(train.row) >= split_count * 2)[0], 
                replace=False,
                size=np.int32(np.floor(fraction * train.shape[0]))
            ).tolist()
        except:
            print(('Not enough users with > {} '
                  'interactions for fraction of {}')\
                  .format(2*k, fraction))
            raise
    else:
        user_index = range(train.shape[0])
        
    train = train.tolil()

    for user in user_index:
        test_ratings = np.random.choice(ratings.getrow(user).indices, 
                                        size=split_count, 
                                        replace=False)
        train[user, test_ratings] = 0.
        # These are just 1.0 right now
        test[user, test_ratings] = ratings[user, test_ratings]
   
    
    # Test and training are truly disjoint
    assert(train.multiply(test).nnz == 0)
    return train.tocsr(), test.tocsr(), user_index

train, test, user_index = train_test_split(likes, 5, fraction=0.2)

"""
	choosing evaluation dataset from training dataset
"""
eval_train = train.copy()
non_eval_users = list(set(range(train.shape[0])) - set(user_index))

eval_train = eval_train.tolil()
for u in non_eval_users:
    eval_train[u, :] = 0.0
eval_train = eval_train.tocsr()

"""
	loading the user profiles
"""
import io
with open('featdlistU_newprofile.txt', 'rb') as handle:
    feat_dlistU = pickle.load(handle)
"""
	loading the item profiles
"""
with open('feat_dlistI_newprofile.txt', 'rb') as handle:
    feat_dlistI = pickle.load(handle)

#vedtorize user_feature
from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer()
user_features = dv.fit_transform( feat_dlistU)
user_features

# Need to hstack item_features
eye = sp.eye(user_features.shape[0], user_features.shape[0]).tocsr()
user_features_concat = sp.hstack((eye, user_features))
user_features_concat = user_features_concat.tocsr().astype(np.float32)
user_features_concat

#vedtorize user_feature
from sklearn.feature_extraction import DictVectorizer
dvI = DictVectorizer()
item_features = dvI.fit_transform( feat_dlistI)
item_features

# Need to hstack item_features
eye1 = sp.eye(item_features.shape[0], item_features.shape[0]).tocsr()
item_features_concat = sp.hstack((eye1, item_features))
item_features_concat = item_features_concat.tocsr().astype(np.float32)
item_features_concat

def objective_wsideinfo(params):
    """
		This function is the objective function for finding the optimum minimum value of loss function 
		for matrix factorization with using just item side information
		but here I should use it, since for finding the hyperparameters for the model with side information
		I need a initial point and this point is better to be the hyperparameter which we found earlier for this model 		
	"""
	"# unpack
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
              item_features=item_features_concat,
              num_threads=6, verbose=True)
    
    #patks = lightfm.evaluation.precision_at_k(model, test,
       #                                       item_features=item_features_concat,
        #                                      train_interactions=None,
         #                                     k=5, num_threads=5)
    auc =lightfm.evaluation.auc_score(model,test,item_features=item_features_concat,train_interactions=None,num_threads=6 )
    mapatk = np.mean(auc)
    # Make negative because we want to _minimize_ objective
    out = -mapatk
    # Weird shit going on
    if np.abs(out + 1) < 0.01 or out < -1.0:
        return 0.0
    else:
        return out

"""
	here the result for hyperparameter for model with just item information is loaded 
	res_fm_auc.x = the best hyperparameter
"""
from skopt import dump,load
res_fm_auc = load('result_forest_minimizer_itemuser_auc.pk1')

"""
	It's the space which we are searching inside it to find the hyperparameters for final model
"""

space = [(1, 160), # epochs
         (10**-3, 1.0, 'log-uniform'), # learning_rate
         (20, 200), # no_components
         (10**-5, 10**-3, 'log-uniform'), # item_alpha
         (0.001, 1., 'log-uniform') # item_scaling
        ]
x0 = res_fm_auc.x.append(1.)
# This typecast is required
user_features = user_features.astype(np.float32)
item_features = item_features.astype(np.float32)
res_fm_useritemfeat = forest_minimize(objective_wsideinfoUI, space, n_calls=20,
                                  x0=x0,
                                  random_state=0,
                                  verbose=True)
"""
	Saving final Hyperparameters
"""
from skopt import dump,load
dump(res_fm_useritemfeat,'result_forest_minimizer_itemuser_auc.pk1')

print('Maximimum auc found: {:6.5f}'.format(-res_fm_useritemfeat.fun))
print('Optimal parameters:')
params = ['epochs', 'learning_rate', 'no_components', 'item_alpha', 'scaling']
for (p, x_) in zip(params, res_fm_useritemfeat.x):
    print('{}: {}'.format(p, x_))
"""
	If we want to visualize the hyperpaarmeters
"""
from skopt.plots import plot_evaluations
_ = plot_evaluations(res_fm_useritemfeat, bins=10, dimensions=['epochs', 'learning_rate', 'no_components', 'item_alpha', 'scaling'])

from skopt.plots import plot_objective

_ = plot_objective(res_fm_useritemfeat, dimensions=['epochs', 'learning_rate', 'no_components', 'item_alpha', 'scaling'])

"""
	Tuning the final model with the tuned hyperparameters
"""
epochs, learning_rate,\
    no_components, item_alpha,\
    scale = res_fm_useritemfeat.x


user_alpha = item_alpha * scale
model = LightFM(loss='warp',
                    random_state=2016,
                    learning_rate=learning_rate,
                    no_components=no_components,
                    user_alpha=user_alpha,
                    item_alpha=item_alpha)

model.fit(likes, epochs=epochs,
              user_features=user_features_concat,item_features=item_features_concat,
              num_threads=6, verbose=True)

"""
	Saving the final model
"""
import pickle
filename = 'lightFM_model4_useritem.sav'
pickle.dump(model, open(filename, 'wb'))

			  