import numpy as np
import pandas as pd
import pickle

from skimage.transform import resize

DIM = 64

# Load original data
df = pd.read_pickle("../data/LSWMD.pkl")

# Load wafer maps with labels
df = df.drop(['waferIndex', 'trianTestLabel', 'lotName'], axis=1)
df['failureNum']=df.failureType
mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
df=df.replace({'failureNum':mapping_type})
df_withlabel = df[(df['failureNum']>=0)]
y = np.array(df_withlabel['failureNum']).astype(np.int)

# Remove abnormal wafer maps with less than 100 dies
df_withlabel = df_withlabel.drop(df_withlabel[df_withlabel['dieSize']<100].index.tolist()).reset_index()

# Binarize and resize wafer maps
X = df_withlabel.waferMap
X_binary = [np.where(x<=1,0,1) for x in X]
X_resize = np.array([resize(x,(DIM,DIM), preserve_range=True, anti_aliasing=False) for x in X_binary])
X_resize = X_resize.reshape(-1,DIM,DIM,1).astype(np.float16)

# Save preprocessed data as pickle files
with open('../data/X_'+str(DIM)+'.pickle', 'wb') as f:
    pickle.dump(X_resize, f, protocol=4)
with open('../data/y.pickle', 'wb') as f:
    pickle.dump(y, f, protocol=4)