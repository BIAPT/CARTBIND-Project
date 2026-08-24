import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import neurokit2 as nk2
import mne
import matplotlib.pyplot as plt
import pickle
from glob import glob
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

time = 'T0'


#Function that perfoms Spatial Feature Aggregation (Only the prefrontal cortex side)
def SFA(dict: dict, epoch: int, metric: str) -> list:
    vector = []
    for ch in prefrontal:
        #Index of the channel
        index = channelToIndex.get(ch) 
        vector.append(dict[epoch][index].get(metric)[0]) #Get the first element from the list
    #assert len(vector) == 31, f"Expected vector of length 5, but got {len(vector)}"
    return vector

#(Only the prefrontal cortex side) (Generate the entire matrix) DON"T USE THIS FUNCTION!!!!
def HFD_prefrontal(file: str) -> np.ndarray:
    epoch = mne.read_epochs(fname=file, preload=True, verbose=False)
    epoch_data = epoch.get_data(copy=True)

    assert isinstance(epoch_data, np.ndarray)

    #Extract the HFD only for prefrontal
    n_epoch, _, _ = epoch_data.shape
    hfd_results = np.zeros((n_epoch, len(prefrontal)))

    """This loop might be wrong (Highly likely wrong)"""
    for e in range(n_epoch):
        for i, ch in enumerate(prefrontal):
            index = channelToIndex.get(ch)
            hfd, _ = nk2.fractal_higuchi(signal=epoch_data[e, index, :], k_max=50, show=False)
            hfd_results[e, i] = hfd

    return hfd_results

#Use this one to generate the vector
def HFD_prefrontal_vector(file: str, epoch_index: int) -> list:
    epoch = mne.read_epochs(fname=file, preload=True, verbose=False)
    epoch_data = epoch.get_data(copy=True)
    assert isinstance(epoch_data, np.ndarray)
    #Extract the HFD only for prefrontal
    hfd_results = []

    for ch in prefrontal:
        index = channelToIndex.get(ch)
        hfd, _ = nk2.fractal_higuchi(signal=epoch_data[epoch_index, index, :], k_max=50)
        hfd_results.append(hfd)
    return hfd_results

#Important file paths
t0_epoch_path = f"/Volumes/NINET/CARTBIND_EEG_data/CARTBIND_CAMH/{time}_epoch"
file_save_path = "/Volumes/NINET/CARTBIND_EEG_data/CARTBIND_CAMH/Features"

ids = [
    '003', '004', '005', '006', '008', '010', '011', '012', '013', '015',
    '018', '019', '021', '023', '024', '026', '029', '030', '031', '032',
    '034', '035', '036', '041', '043', '044', '045', '048', '051', '052',
    '054', '056', '058', '059', '061', '062', '063', '064', '067', '068',
    '069', '072', '073', '074', '075', '077', '078', '082', '083', '087',
    '088', '089', '091', '093', '094', '095', '096', '097', '098', '099',
    '102', '106', '112', '119', '120', '122', '123', '125', '127', '130', '136'
]
t0_complexity = f"/Volumes/NINET/CARTBIND_EEG_data/CARTBIND_CAMH/Features/{time}"

#Categories the dictionary has
categories = ['Entropy', 'Complexity']

# prefrontal = ['T7', 'T8', 'TP10', 'TP7', 'TP8', 'TP9']
frontal_channels = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Fz']
frontal_region_channels = [
    'AF3', 'AF4', 
    'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Fz', 
    'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCz', 
    'FT7', 'FT8', 
    'Fp1', 'Fp2', 'Fpz'
]

n_epoch = 30
n_channels = 6
n_complexities = 6

ch_names = pd.read_pickle(f"{t0_complexity}/004_REST_EC_{time}_complex_results.pkl")['Ch_names']
print(f"Ch_names: {ch_names}\nLength: {len(ch_names)}")


feature_collection_t0 = []
feature_collection_t1 = []
t0_patient_files = []


for id in ids:
    feature_tensor_t0 = []
    feature_tensor_t1 = []
    
    t0_data = pd.read_pickle(f"{t0_complexity}/{id}_REST_EC_{time}_complex_results.pkl")
    ch_list = t0_data['Ch_names']
    temporal = ['T7', 'T8', 
        'TP9', 'TP10', 
        'FT7', 'FT8', 
        'TP7', 'TP8']

    
    prefrontal = list(set(ch_list) & set(frontal_channels))
    channelToIndex = {ch_name : index for index, ch_name in 
                  enumerate(pd.read_pickle(f"{t0_complexity}/{id}_REST_EC_{time}_complex_results.pkl")['Ch_names'])}

    t0_params = t0_data['Parameters']
    for cat in categories:
        """
        The line below is to be removed
        For now no entropy will be part of the feature
        """
        if cat == 'Entropy':
            continue
        epoch_t0 = t0_params[cat]
        
        metrics = ['HC_LMC', 'HC_PEn']
        epoch_num = 30 #Some recordings have more than 18 epochs (Use the min # of epochs)

        #Define the matrix
        mat_t0 = []
        mat_t1 = []

        for e in range(len(t0_params['Entropy'].keys())):
            epoch_deltas = []
            matrix_t0 = []
            matrix_t1 = []
            for m in metrics:
                prefront_epoch_t0 = SFA(dict=epoch_t0, epoch=e, metric=m)
                #prefront_epoch_t1 = SFA(dict=epoch_t1, epoch=e, metric=m)
                #Create [key : value] metric : ndarray or metric : tensor?
                #Average the FDS per channel
                matrix_t0.append(prefront_epoch_t0)
                #matrix_t1.append(prefront_epoch_t1)
            
            
            #Create 2D matrix
            mat_t0 = np.column_stack(matrix_t0) 
            feature_tensor_t0.append(mat_t0)
            #feature_tensor_t1.append(mat_t1)
    print("Feature tensor shape: ", np.array(feature_tensor_t0).shape)
    feature_tensor_t0 = np.mean(feature_tensor_t0, axis=(0,1))
    print("Feature ten dim: ", np.array(feature_tensor_t0).shape)
    feature_collection_t0.append(np.stack(feature_tensor_t0, axis=0))
    #feature_collection_t1.append(np.stack(feature_tensor_t1, axis=0))
    
    """saving the tensors"""
    print(f"Done ID: {id}")
    print(np.array(feature_collection_t0).shape)

np.save(f"{file_save_path}/{time}_dataset/{time}_feature_collection_frontal_CARTBIND.npy", np.array(feature_collection_t0))
# np.save("Dataset/train/T1_feature_collection_occipital.npy", np.array(feature_collection_t1))
print("Done")

label = [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 
        0, 1, 1, 0, 0, 0, 0, 0, 1, 1,
        0, 1, 1, 1, 1, 1, 0, 0, 1, 0,
        1, 1, 0, 1, 1, 0, 1, 0, 0, 0]


0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
1, 0, 1, 1, 0, 1, 0, 0, 0, 1,
0, 0, 1, 1, 0, 0, 1, 0, 0, 0,
0, 1, 0, 1, 1, 0, 0, 0, 0, 0,
0, 1, 0, 1, 0, 0, 0, 1, 0, 0,
0, 1, 0, 0, 1, 0, 0, 0, 1, 0,
1, 0, 0, 0, 0, 0, 1, 1, 0, 0,
0