import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import multiprocessing

def brain_region_dataset(path, time, region=None):
    feature = np.load(f"{path}/{time}_dataset/{time}_feature_collection_temporal_CARTBIND.npy")
    # n_samples, n_epochs, n_channels, n_features = feature.shape
    # feature_reshape = np.mean(feature, axis=(1,2))
    return feature

#label_path = "/Users/andrewlee/BIAPT_Lab/PICU_Criticality_Prognosis/scripts/variable_analysis/rTMS_labels.xlsx"
feature_path = "/Volumes/NINET/CARTBIND_EEG_data/CARTBIND_UBC/Features"

"""No label below commented out"""
# excel_df = pd.read_excel(io=label_path, sheet_name=0) #Dataframe
# excel_df.index = excel_df.index+2
# """Extract the labels for CNN Lab (for now)"""
# #print(excel_df)
# excel_CNN = excel_df[:47]
# indices = [12, 19, 20, 22, 26, 30, 34, 35, 40, 44, 45]
# excel_CNN = excel_CNN.drop(index=indices) #Non-overlap dropped
# # print(excel_CNN)

# overlap_indices = excel_CNN.index
# male_count = 0
# female_count = 0
# clinical_response = excel_CNN["Response"][overlap_indices]
# labels = []
# for response in clinical_response:
#     if response == "yes":
#         labels.append(int(1))
#     elif response == "no":
#         labels.append(int(0))
# assert len(labels) == 36, f"Returned size: {len(labels)}"

#T0 and T1 path
region =["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
# T0_feature_frontal = brain_region_dataset(path=feature_path, region="frontal", time="T0")
#T0_feature_central = brain_region_dataset(path=feature_path, region="central", time="T0")
# T0_feature_parietal = brain_region_dataset(path=feature_path, region="parietal", time="T0")
# T0_feature_occipital = brain_region_dataset(path=feature_path, region="occipital", time="T0")

# T1_feature_frontal = brain_region_dataset(path=feature_path, region="frontal", time="T1")
# T1_feature_central = brain_region_dataset(path=feature_path, region="central", time="T1")
# T1_feature_parietal = brain_region_dataset(path=feature_path, region="parietal", time="T1")
# T1_feature_occipital = brain_region_dataset(path=feature_path, region="occipital", time="T1")
# # --- 2. Create the 3D Plot ---
#fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': '3d'})

#ax.scatter(T1_feature_parietal[:,0], T1_feature_parietal[:,1], T1_feature_parietal[:,2], color='blue', label='Frontal (T0)', s=40, edgecolors='k')
#ax.scatter(T0_feature_central[:,0], T0_feature_central[:,1], T0_feature_central[:,2], color=region[1], label='Central (T0)', s=40, edgecolors='k')
# ax.scatter(T0_feature_parietal[:,0], T0_feature_parietal[:,1], T0_feature_parietal[:,2], color=region[2], label='Parietal (T0)', s=40, edgecolors='k')
# ax.scatter(T0_feature_occipital[:,0], T0_feature_occipital[:,1], T0_feature_occipital[:,2], color=region[3], label='Occipital (T0)', s=40, edgecolors='k')
# """Responder/Non-responder group"""
# ax.scatter(T1_feature_frontal[:,0], T1_feature_frontal[:,1], T1_feature_frontal[:,2], color=region[1], label='Frontal (T1)', s=40, edgecolors='k')
# ax.scatter(T1_feature_frontal[:,0], T1_feature_frontal[:,1], T1_feature_frontal[:,2], c=labels, cmap='coolwarm', label='Frontal (T1)', s=40, edgecolors='k')

"""Extract features"""
T0_feature = brain_region_dataset(path=feature_path, time="T0")
T1_feature = brain_region_dataset(path=feature_path, time="T1")
T2_feature = brain_region_dataset(path=feature_path, time="T2")


"""Generate the direction of the change"""
regions = {
    "All": (T0_feature, T1_feature, "orange"),
    # "Frontal": (T0_feature_frontal, T1_feature_frontal, "#1f77b4"),
    # "Central": (T0_feature_central, T1_feature_frontal, "#ff7f0e"),
    # "Parietal": (T0_feature_parietal, T1_feature_parietal, "#2ca02c"),
    # "Occipital": (T0_feature_occipital, T1_feature_occipital, "#9467bd")
}

# for region, (T0, T1, color) in regions.items():
#     #T0
#     x0 = T0[:,0]
#     y0 = T0[:,1]
#     # z0 = T0[:,2]
#     #T1
#     x1 = T1[:,0]
#     y1 = T1[:,1]
#     # z1 = T1[:,2]

#     z0 = np.array([0]*27)
#     z1 = np.array([0]*27)
#     plt.scatter(x0, y0,
#             #    c=1,
#             #    cmap='coolwarm',
#                color="red",
#                alpha=0.6,
#                label=f"{region} T0")
#     # Plot T1 points
#     plt.scatter(x1, y1,
#             #    c=1,
#             #    cmap='coolwarm',
#                color="blue",
#                marker='^',
#                alpha=0.6,
#                label=f"{region} T1")
#     # Draw arrows
#     # plt.quiver(
#     #     x0, y0,          # start points
#     #     x1-x0,                # dx
#     #     y1-y0,                # dy
#     #                    # dz
#     #     color='black',
#     #     alpha=0.4,
#     #     linewidth=1
#     # )


# plt.figure(figsize=(6, 4))
# plt.scatter(x_coord_T0, y_coord_T0, marker='o', linestyle='-')
# plt.scatter(x_coord_T1, y_coord_T1, marker='o', linestyle='-')
# plt.scatter(T2_feature[:,1], T2_feature[:,0], marker='o', linestyle='-')


# # --- 3. Labels and Global Constraints ---
# plt.xlabel('Permutation Entropy ($PEn$)', fontsize=11, labelpad=10)
# # plt.set_zlabel('Fisher Information ($FI$)', fontsize=11, labelpad=10)
# plt.ylabel('Statistical Complexity ($LMC$)', fontsize=11, labelpad=10)
# plt.title('2D Information-Theoretic Complexity Space', fontsize=14, pad=20)

# # Set axes boundaries based on mathematical limits (for embedding dimension D=3)
# plt.xlim(0.5, 1.0)
# plt.ylim(0, 0.5)
# # plt.set_zlim(0, 0.5) 

# # Adjust viewing angle for optimal perspective of the 3D curve
# # plt.view_init(elev=25, azim=-135)

# plt.legend(loc='upper left', bbox_to_anchor=(0.1, 0.85))
# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt

resp_idx = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 19, 20, 22, 23, 24, 25, 26, 29, 31, 32, 34, 35, 37]

non_resp_idx = [0, 4, 11, 14, 15, 16, 17, 18, 21, 27, 28, 30, 33, 36, 38, 39, 40]

plt.figure(1)
#Drop the resp/non_resp
x_coord_T0 = T0_feature[:,1]
y_coord_T0 = T0_feature[:,0]

x_coord_T1 = T1_feature[:,1]
y_coord_T1 = T1_feature[:,0]

x_coord_T2 = T2_feature[:, 1]
y_coord_T2 = T2_feature[:, 0]

# plt.scatter(x_coord_T0, y_coord_T0, label='T0', marker='o')
# plt.scatter(x_coord_T1, y_coord_T1, label='T1', marker='o')
# plt.scatter(T2_feature[:,1], T2_feature[:,0], label='T2', marker='o')

#Change in to not in for non_resp group
resp_x_T0 = [row for i, row in enumerate(x_coord_T0) if i not in resp_idx]
resp_y_T0 = [row for i, row in enumerate(y_coord_T0) if i not in resp_idx]

resp_x_T1 = [row for i, row in enumerate(x_coord_T1) if i not in resp_idx]
resp_y_T1 = [row for i, row in enumerate(y_coord_T1) if i not in resp_idx]

resp_x_T2 = [row for i, row in enumerate(x_coord_T2) if i not in resp_idx]
resp_y_T2 = [row for i, row in enumerate(y_coord_T2) if i not in resp_idx]

x_T0 = np.mean(resp_x_T0,axis=0)
y_T0 = np.mean(resp_y_T0, axis=0)

x_T1 =np.mean(resp_x_T1,axis=0)
y_T1 = np.mean(resp_y_T1, axis=0)

x_T2 = np.mean(resp_x_T2,axis=0)
y_T2=np.mean(resp_y_T2,axis=0)

plt.scatter(np.mean(resp_x_T0,axis=0), np.mean(resp_y_T0, axis=0), label='T0', marker='o', facecolors='none', edgecolors=region[3])
plt.scatter(np.mean(resp_x_T1,axis=0), np.mean(resp_y_T1, axis=0), label='T1', marker='^', color=region[2])
plt.scatter(np.mean(resp_x_T2,axis=0), np.mean(resp_y_T2,axis=0), label='T2', marker='o', color=region[0])

plt.annotate('', 
                 xy=(x_T1, y_T1),      # Arrow head target
                 xytext=(x_T0, y_T0),  # Arrow tail start
                 arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, alpha=0.6))
    
    # Arrow from T1 -> T2
plt.annotate('', 
                 xy=(x_T2, y_T2),      # Arrow head target
                 xytext=(x_T1, y_T1),  # Arrow tail start
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.5, alpha=0.6))
plt.xlabel("PEn")
plt.ylabel("LMC (Statistical complexity)")
plt.title("Trend in Non-Responder Group (Temporal)")
plt.legend()
# plt.savefig("non_resp_trend_temporal_UBC.png", dpi=300)

# for i in range(len(x_coord_T0)):
#     if i in resp_idx:
#         continue
#     # Arrow from T0 -> T1
#     plt.annotate('', 
#                  xy=(x_coord_T1[i], y_coord_T1[i]),      # Arrow head target
#                  xytext=(x_coord_T0[i], y_coord_T0[i]),  # Arrow tail start
#                  arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, alpha=0.6))
    
#     # Arrow from T1 -> T2
#     plt.annotate('', 
#                  xy=(x_coord_T2[i], y_coord_T2[i]),      # Arrow head target
#                  xytext=(x_coord_T1[i], y_coord_T1[i]),  # Arrow tail start
#                  arrowprops=dict(arrowstyle="->", color="black", lw=1.5, alpha=0.6))

# plt.legend()



# plt.figure(2)

# resp_x_T0_2 = [row for i, row in enumerate(x_coord_T0) if i in resp_idx]
# resp_y_T0_2 = [row for i, row in enumerate(y_coord_T0) if i in resp_idx]

# resp_x_T1_2 = [row for i, row in enumerate(x_coord_T1) if i in resp_idx]
# resp_y_T1_2 = [row for i, row in enumerate(y_coord_T1) if i in resp_idx]

# resp_x_T2_2 = [row for i, row in enumerate(x_coord_T2) if i in resp_idx]
# resp_y_T2_2 = [row for i, row in enumerate(y_coord_T2) if i in resp_idx]

# plt.scatter(resp_x_T0_2, resp_y_T0_2, label='T0', marker='o')
# plt.scatter(resp_x_T1_2, resp_y_T1_2, label='T1', marker='o')
# plt.scatter(resp_x_T2_2, resp_y_T2_2, label='T2', marker='o')

# for i in range(len(x_coord_T0)):
#     if i in non_resp_idx:
#         continue
#     # Arrow from T0 -> T1
#     plt.annotate('', 
#                  xy=(x_coord_T1[i], y_coord_T1[i]),      # Arrow head target
#                  xytext=(x_coord_T0[i], y_coord_T0[i]),  # Arrow tail start
#                  arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, alpha=0.6))
    
#     # Arrow from T1 -> T2
#     plt.annotate('', 
#                  xy=(x_coord_T2[i], y_coord_T2[i]),      # Arrow head target
#                  xytext=(x_coord_T1[i], y_coord_T1[i]),  # Arrow tail start
#                  arrowprops=dict(arrowstyle="->", color="black", lw=1.5, alpha=0.6))

# plt.legend()


# plt.figure(3)
resp_x_T0_2 = [row for i, row in enumerate(x_coord_T0) if i in resp_idx]
resp_y_T0_2 = [row for i, row in enumerate(y_coord_T0) if i in resp_idx]

resp_x_T1_2 = [row for i, row in enumerate(x_coord_T1) if i in resp_idx]
resp_y_T1_2 = [row for i, row in enumerate(y_coord_T1) if i in resp_idx]

resp_x_T2_2 = [row for i, row in enumerate(x_coord_T2) if i in resp_idx]
resp_y_T2_2 = [row for i, row in enumerate(y_coord_T2) if i in resp_idx]

x_T0_2 = np.mean(resp_x_T0_2,axis=0)
y_T0_2 = np.mean(resp_y_T0_2, axis=0)

x_T1_2 =np.mean(resp_x_T1_2,axis=0)
y_T1_2 = np.mean(resp_y_T1_2, axis=0)

x_T2_2 = np.mean(resp_x_T2_2,axis=0)
y_T2_2=np.mean(resp_y_T2_2,axis=0)

plt.scatter(np.mean(resp_x_T0_2,axis=0), np.mean(resp_y_T0_2, axis=0), label='T0', marker='o', facecolors='none', edgecolors=region[3])
plt.scatter(np.mean(resp_x_T1_2,axis=0), np.mean(resp_y_T1_2, axis=0), label='T1', marker='^', color=region[2])
plt.scatter(np.mean(resp_x_T2_2,axis=0), np.mean(resp_y_T2_2,axis=0), label='T2', marker='o', color=region[0])

plt.annotate('', 
                 xy=(x_T1_2, y_T1_2),      # Arrow head target
                 xytext=(x_T0_2, y_T0_2),  # Arrow tail start
                 arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, alpha=0.6))
    
    # Arrow from T1 -> T2
plt.annotate('', 
                 xy=(x_T2_2, y_T2_2),      # Arrow head target
                 xytext=(x_T1_2, y_T1_2),  # Arrow tail start
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.5, alpha=0.6))
plt.xlabel("PEn")
plt.ylabel("LMC (Statistical complexity")
plt.title("Trend in Responder Group (Temporal)")
# plt.savefig("resp_trend_temporal_UBC.png", dpi=300)
plt.show()