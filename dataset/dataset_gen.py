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


"""
Classify the clustering of data with no labels
- Perform unsupervised learning (K-mean)
- To avoid curse of dimensionality in K-mean apply Spatial feature aggregation
  followed by PCA for further feature reduction
- Reason why dimension reduction is required: for high dimension in statistics, 
  # of feature >> # of samples and for high dimension (infinite volume)
  which results in K-mean that relies on Euclidean distance to be roughly
  equidistance as points in n-dim are sparse and appear as uniform cloud
  
- For n-dimensional feature (n number of complexity + HFD, n channle, n epoch)
  reduce the dimension and perform K-mean
- Generate the visual for the classified group (Responder and non-responder)
- Compare with the True label later
- If High accuracy/similarity between True and learned label
  this indicates some meaningful relationship
"""

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
# t0_patient_path = "/Volumes/NINET/CARTBIND_EEG_data/CARTBIND_UBC/Features/T0"
t0_epoch_path = "/Volumes/NINET/CARTBIND_EEG_data/CARTBIND_UBC/T0_epoch"
#t1_patient_path = "/Users/andrewlee/BIAPT_Lab/EEG_Preprocessing/EEG_dataset/CNN_Lab/Patients/T1_filtered_epoch"
file_save_path = "/Volumes/NINET/CARTBIND_EEG_data/CARTBIND_UBC/Features"


ids = ['0001', '0003', '0004', '0005', '0007', '0008', '0009', '0010', '0011', '0012', 
       '0015', '0017', '0018', '0022', '0025', '0027', '0034', '0036', '0037', '0038', 
       '0040', '0044', '0045', '0047', '0050', '0051', '0052']



# #For testing dataset
# ids_test = [
#     727, 728, 802, 826, 827, 829, 843, 850, 869, 876, 885, 911
# ]

# ids = [
#     "050", "051", "052", "053", "055", "056", "057", "058", "059", "061", "062"
# ]

#For testing purposes (Patient ID: 731)
t0_complexity = "/Volumes/NINET/CARTBIND_EEG_data/CARTBIND_UBC/Features/T0"
# t1_complexity = "/Users/andrewlee/BIAPT_Lab/PICU_Criticality_Prognosis/ANALYSIS/CNN_Lab/Patients/T1"

#Categories the dictionary has
categories = ['Entropy', 'Complexity']
num_channels = 61  # Based on your previous output (hardcoded may need to be changed later)

prefrontal = ['T7', 'T8', 'TP10', 'TP7', 'TP8', 'TP9']


"""Commented out for now"""
# channelToIndex = {ch_name : index for index, ch_name in 
#                   enumerate(pd.read_pickle(f"{t0_complexity}/0005_REST_EC_T0_complex_results.pkl")['Ch_names'])}

print(pd.read_pickle(f"{t0_complexity}/0005_REST_EC_T0_complex_results.pkl")['Ch_names'])

print("1")
#First [0] means Epoch level and the second [0] is the channel
# print(t0_params['Complexity'][0][0]['HC_LMC'][1])

#Create new ndim array that can be used for feature mapping
#/T0_731_complex_results.pkl" (Example of the file type)

"""From here"""
"""Creating the tensor (n_epoch, n_channels, n_features)"""
"""For now, use ['HC_LMC', 'HC_PEn', 'HC_FI', 'LZC', 'PLZC'] and HFD for the feature"""

#Hard code the n_epoch for now
n_epoch = 30
n_channels = 6
n_complexities = 6

# LMC_matrix = 
# PEn_matrix = 
# FI_matrix = 
# LZC_matrix = 
# PLZC_matrix = 

feature_collection_t0 = []
feature_collection_t1 = []
t0_patient_files = []


for id in ids:
    # print(file)
    feature_tensor_t0 = []
    feature_tensor_t1 = []
    #Skip non overlapping ids
    # id = os.path.basename(file).split("_")[1]
    # if str(id) not in ids:
    #     continue
    #Get the .pkl data
    t0_data = pd.read_pickle(f"{t0_complexity}/{id}_REST_EC_T0_complex_results.pkl")
    channelToIndex = {ch_name : index for index, ch_name in 
                  enumerate(pd.read_pickle(f"{t0_complexity}/{id}_REST_EC_T0_complex_results.pkl")['Ch_names'])}
    #t1_data = pd.read_pickle(f"{t1_complexity}/T1_{id}_complex_results.pkl")
    t0_params = t0_data['Parameters']
    #t1_params = t1_data['Parameters']

    for cat in categories:
        """
        The line below is to be removed
        For now no entropy will be part of the feature
        """
        if cat == 'Entropy':
            continue
        epoch_t0 = t0_params[cat]
        #epoch_t1 = t1_params[cat]
        
        # if not epoch_t0 or not epoch_t1:
        #     print("No dictionary exists")
        #     continue
        metrics = epoch_t0[0][0].keys()
        epoch_num = 30 #Some recordings have more than 18 epochs (Use the min # of epochs)

        #Define the matrix
        mat_t0 = []
        mat_t1 = []

        for e in range(epoch_num):
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
            
            #Append the HFD vector
            HFD_t0_vec = HFD_prefrontal_vector(file=f"{t0_epoch_path}/CBN02_{id}_REST_EC_T0_UBC_epo.fif", epoch_index=e)
            #HFD_t1_vec = HFD_prefrontal_vector(file=f"{t1_patient_path}/Rest_{id}_filtered-epo.fif", epoch_index=e)
            
            #Append the last vector
            matrix_t0.append(HFD_t0_vec)
            #matrix_t1.append(HFD_t1_vec)
            
            #Create 2D matrix
            mat_t0 = np.column_stack(matrix_t0) 
            #mat_t1 = np.column_stack(matrix_t1) 

            #print(mat_t0.shape)
            assert mat_t0.shape == (n_channels, n_complexities)
            #Append the matrix (Creating tensor channel size == n_epoch)
            feature_tensor_t0.append(mat_t0)
            #feature_tensor_t1.append(mat_t1)

    #Now stack all of them
    #print(np.stack(feature_tensor_t0, axis=0).shape)
    assert np.stack(feature_tensor_t0, axis=0).shape == (n_epoch, n_channels, n_complexities)
    feature_collection_t0.append(np.stack(feature_tensor_t0, axis=0))
    #feature_collection_t1.append(np.stack(feature_tensor_t1, axis=0))
    
    """saving the tensors"""
    print(f"Done ID: {id}")
    print(np.array(feature_collection_t0).shape)

np.save(f"{file_save_path}/T0_dataset/T0_feature_collection_temporal_CARTBIND.npy", np.array(feature_collection_t0))
# np.save("Dataset/train/T1_feature_collection_occipital.npy", np.array(feature_collection_t1))
print("Done")

label = [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 
        0, 1, 1, 0, 0, 0, 0, 0, 1, 1,
        0, 1, 1, 1, 1, 1, 0, 0, 1, 0,
        1, 1, 0, 1, 1, 0, 1, 0, 0, 0]