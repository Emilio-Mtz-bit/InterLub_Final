import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('../data/datos_onehotenocder.csv')
data.info()
def Cos_Model(X_data, feature_vector):
    '''
    This function calculates the cosine similarity of a certain vector with a database and return the idx and the value of the first 5 higher in value
    '''
    X_data_array = X_data.to_numpy()
    vector_2d = np.array(feature_vector).reshape(1, -1)
    vector_2d = np.nan_to_num(vector_2d, nan=0.0)
    X_data_array = np.nan_to_num(X_data_array, nan=0.0)
    simil = cosine_similarity(X_data_array, vector_2d)
    
    idx = np.argsort(simil.flatten())
    
    idx_descendente = idx[::-1]
    
    idx_rank = idx_descendente[:5]
    scores_rank = simil[idx_descendente]
    
    return idx_rank, scores_rank, idx