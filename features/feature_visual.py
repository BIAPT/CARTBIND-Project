
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import neurokit2 as nk2
import mne
import matplotlib.pyplot as plt
import matplotlib
import pickle
from glob import glob
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

"""Testing code
    Not used anymore
"""

file_save_path = "/Volumes/NINET/CARTBIND_EEG_data/CARTBIND_UBC/Features"

n_epoch = 30
n_channels = 61
n_complexities = 6

feature_collection_t0 = np.load(f"{file_save_path}/T0_dataset/T0_feature_collection_temporal_CARTBIND.npy")
feature_collection_t1 = np.load(f"{file_save_path}/T1_dataset/T1_feature_collection_temporal_CARTBIND.npy")
feature_collection_t2 = np.load(f"{file_save_path}/T2_dataset/T2_feature_collection_temporal_CARTBIND.npy")

"""Perform Global Scaling"""
#Create a collection of covariance matrix
# assert np.array(feature_collection_t0).shape == np.array(feature_collection_t1).shape
#These collection should contain scaled mean of the tensors
cov_mat_collection_t0 = []
cov_mat_collection_t1 = []
#Collection of scaled datasets
scaled_features_t0 = []
scaled_features_t1 = []

#Use global scale for calculating cov matrices (Using Baseline T0)
global_mean = np.array(feature_collection_t0).mean(axis=(0,1,2))
global_std = np.array(feature_collection_t0).std(axis=(0,1,2), ddof=0)
print(f"Global mean: {global_mean}")
print(f"Global std: {global_std}")


#Extracing the cov_mat logic
for tensor_t0 in (feature_collection_t0):
    cov_mat_tensor_t0 = []
    #cov_mat_tensor_t1 = []
    scaled_tensor_t0 = []
    #scaled_tensor_t1 = []
    #Scale the cov matrix to avoid max feature from dominating the calculation
    cov_mat_scaled_t0 = np.cov(np.array(np.mean(tensor_t0, axis=1)).reshape(-1, n_complexities), rowvar=False, ddof=0)
    #cov_mat_scaled_t1 = np.cov(np.array(np.mean(tensor_t1, axis=1)).reshape(-1, n_complexities), rowvar=False, ddof=0)
    #Append the result
    cov_mat_tensor_t0.append(cov_mat_scaled_t0)
    #cov_mat_tensor_t1.append(cov_mat_scaled_t1)
    for mat_t0 in (tensor_t0):
        #Global adjustment applied (scaling)
        mat_scaled_t0 = (mat_t0 - global_mean)/global_std
        #mat_scaled_t1 = (mat_t1 - global_mean)/global_std
        scaled_tensor_t0.append(mat_scaled_t0)
        #scaled_tensor_t1.append(mat_scaled_t1)

    #Compute the mean across all epoch (per tensor)
    print(f"Scaled cov mat: {cov_mat_scaled_t0.shape}")
    cov_mat_collection_t0.append(np.mean(np.array(cov_mat_tensor_t0), axis=0))
    print(f"Collection shape: {np.array(cov_mat_collection_t0).shape}")
    #cov_mat_collection_t1.append(np.mean(np.array(cov_mat_tensor_t1), axis=0))
    #Compute the same for scaled mat
    scaled_features_t0.append(np.mean(np.array(scaled_tensor_t0), axis=(0,1)))
    #scaled_features_t1.append(np.mean(np.array(scaled_tensor_t1), axis=(0,1)))


"""For visual"""
# #Make sure the matrix is square
# print(np.array(cov_mat_collection_t0).shape)

# # Labels used for covariance matrix
# labels = ['HC_LMC', 'HC_PEn', 'HC_FI', 'LZC', 'PLZC', 'HFD']

# # 3. Create the heatmap
# avg_cov_mat = cov_mat_collection_t0[1]
# plt.figure(figsize=(10, 8))
# sns.heatmap(avg_cov_mat, annot=True, fmt=".2f", cmap='coolwarm', 
#             xticklabels=labels, yticklabels=labels, center=0)

# avg_cov_mat_t1 = cov_mat_collection_t1[1]
# plt.figure(figsize=(10, 8))
# sns.heatmap(avg_cov_mat_t1, annot=True, fmt=".2f", cmap='coolwarm', 
#             xticklabels=labels, yticklabels=labels, center=0)
# plt.show()
"""Calculate Principal Components"""
print(f"scaled feature size: {np.array(scaled_features_t0).shape}")
print(np.array(cov_mat_collection_t0).shape)
mean_cov_mat_t0 = np.mean(cov_mat_collection_t0, axis=0)
print(np.array(mean_cov_mat_t0).shape)
print(mean_cov_mat_t0)
assert np.array(mean_cov_mat_t0).shape == (6,6)

#Eigenvectors in column order
eigenVal, eigenVec = np.linalg.eigh(mean_cov_mat_t0)

#Sort it in descending order 
#(np.argsort sorts in ascending order and returns the indices but ::-1 reverses the list)
index = np.argsort(eigenVal)[::-1]

#Reorder eigenvectors in the order of the sorted indices
pca_weights = eigenVec[:,index]
#Pick top 3 principal components
pc1 = pca_weights[:,0]
pc2 = pca_weights[:,1]
pc3 = pca_weights[:,2]

print(f"PC1 captures: {eigenVal[index[0]]/np.sum(eigenVal)}")
print(f"PC2 captures: {eigenVal[index[1]]/np.sum(eigenVal)}")
print(f"PC3 captures: {eigenVal[index[2]]/np.sum(eigenVal)}")

"""Project it onto the PCA Dimension"""
trans_mat = np.stack([pc1,pc2,pc3], axis=1)
print(f"weights: {trans_mat.shape}")
assert np.array_equal(trans_mat, pca_weights[:,:3])

"""Treat each epoch as independent point"""
scaled_feature_col_t0 = []
for tesnor_t0 in feature_collection_t0:
    scaled_tensor_t0 = []
    for m_t0 in tensor_t0:
        mat_scaled_t0 = (m_t0 - global_mean)/global_std
        scaled_tensor_t0.append(np.array(mat_scaled_t0)@trans_mat)
    scaled_feature_col_t0.append(scaled_tensor_t0)


# #Plotting
# fig = plt.figure(figsize=(8,6))
# axis = fig.add_subplot(111, projection='3d')
# #["Fp1", "Fp2", "F3", "F4", "Fpz"]
# colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# for tensor in scaled_feature_col_diff:
#     #print(f"Features shape: {np.array(features).shape}")
#     vector = np.mean(tensor, axis=0)
#     axis.scatter(
#         vector[:,0],
#         vector[:,1],
#         vector[:,2],
#         color='#2ca02c',
#         alpha=0.6
#     )

# axis.set_xlabel("PC1")
# axis.set_ylabel("PC2")
# axis.set_xlabel("PC3")
# plt.suptitle(f"Baseline scaling of T0 and T1 difference in PCA dimension Averaged Prefrontal")
# plt.savefig(f"Diff_F3_all_epoch.png", dpi=300)

#Plotting
fig = plt.figure(figsize=(8,6))
axis = fig.add_subplot(111, projection='3d')
#["Fp1", "Fp2", "F3", "F4", "Fpz"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

for mat in np.mean(scaled_feature_col_t0, axis=2):
    #print(f"Features shape: {np.array(features).shape}")
    for vec in mat:
        # for i, vector in enumerate(mat):
        axis.scatter(
            vec[0],
            vec[1],
            vec[2],
            color=colors[1],
            alpha=0.6
        )
axis.set_xlabel("PC1")
axis.set_ylabel("PC2")
axis.set_zlabel("PC3")
plt.suptitle(f"Baseline scaling of T0 and T1 difference in PCA dimension F3 Channel")
#plt.savefig(f"Diff_average_prefrontal_all_epoch.png", dpi=300)
plt.show()


# """Plotting the difference"""
# scaled_diff = np.array(scaled_features_t0) - np.array(scaled_features_t1)
# trans_diff = []
# for mat in scaled_diff:
#     trans_diff.append((mat)@trans_mat)

# fig = plt.figure(figsize=(8,6))
# axis = fig.add_subplot(111, projection='3d')
# #["Fp1", "Fp2", "F3", "F4", "Fpz"]
# colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
# print(np.array(trans_diff).shape)
# for feature in trans_diff:
#     #print(f"Features shape: {np.array(features).shape}")
#     # for i, feature in enumerate(features):
#         #print(f"Feature shape: {np.array(feature).shape}")
#     axis.scatter(
#         feature[0],
#         feature[1],
#         feature[2],
#         color=colors[1],
#         alpha=0.6
#     )

# axis.set_xlabel("PC1")
# axis.set_ylabel("PC2")
# axis.set_zlabel("PC3")
# plt.suptitle(f"Baseline scaling of T0 and T1 difference in PCA dimension F3 (Epoch averaged)")
# plt.show()