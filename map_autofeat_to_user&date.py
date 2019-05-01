import os
import json
import numpy as np


# ----------------------------
# Modify the following vars to point to the right locations of 
# features_path, orig_split_path, and cv_split_path

fold = 0
features_path = '/home/bl41/Projects/scalableh/models/encoders/LR+0.01|BS+100|OPT+adam|GC+5|NOI+0.05|LAT+12|SR+8HZ|UNSUPV|LocaGausAutoencoderWithBnSR+well+mse+dcorr+/f%d/auto_features_corr_ep9.npy'%fold
orig_split_path = '/home/bl41/Projects/scalableh/data/indx-semi_sort_normdata_swwb-udep-seed_42-intersect.json'
cv_split_path = '/home/bl41/Projects/scalableh/data/indx-semi_sort_normdata_swwb_v2-udep-seed_42-intersect.json'

'''
Structure of the resulted organized_features:
organized_features = {
    'train':{
        'index':    [], # a list of id-date code, mapping to features and labels in the same set
        'features': np.array, # of size (xxx, 12)
        'labels':   np.array  # of size (xxx, 10)
    },
    'val':{ # similar as train set
        'index':    [],
        'features': np.array,
        'labels':   np.array 
    },
    test:{ # similar
        'index':    [], 
        'features': np.array, 
        'labels':   np.array 
    }
}
'''
# ----------------------------


phases = ['train','val','test']

with open(orig_split_path, 'r') as handle:
    orig_split = json.loads(handle.read())
    id_date_index = np.concatenate([orig_split[p] for p in phases]).tolist()

with open(cv_split_path, 'r') as handle:
    cv_split = json.loads(handle.read())['f%d'%fold]
    
features = np.load(features_path).item()
X_all = np.concatenate([features[p][0] for p in phases])
y_all = np.concatenate([features[p][1] for p in phases])
print(X_all.shape, y_all.shape)

organized_features = dict.fromkeys(phases, None)
for p in phases:
    organized_features[p] = {'index':[], 'features':[], 'labels':[]}
    organized_features[p]['index'] = cv_split[p]
    organized_features[p]['features'] = np.empty((len(organized_features[p]['index']),*X_all.shape[1:]))
    organized_features[p]['labels'] = np.empty((len(organized_features[p]['index']),*y_all.shape[1:]))
    for i,item in enumerate(organized_features[p]['index']):
        found_orig_index = id_date_index.index(item)
        organized_features[p]['features'][i,...] = X_all[found_orig_index,...]
        organized_features[p]['labels'][i,...] = y_all[found_orig_index,...]

print('Old:', *['num %s = %d;'%(p,len(orig_split[p])) for p in phases])
print('New(REAL):', *['num %s = %d;'%(p,len(organized_features[p]['features'])) for p in phases])
