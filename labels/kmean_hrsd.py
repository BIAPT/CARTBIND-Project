#running kmean clustering (3 clusters) in 1d space
#formula used for values in 1d space (hrsd_total_t30 - hrsd_total_b1)/(hrsd_total_b1)*100
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

patient_summary = pd.read_excel('CARTBIND_for_Andrew_labels.xlsx', sheet_name='HRSD17')
#find the ID with no hrsd_total_b1 & t30 -> remove them
# no_hrsd_b1 = [hrsd for hrsd in patient_summary['hrsd_total_b1'] if hrsd == 'NaN']
patient_summary = patient_summary.dropna()
#extract b1 & t30
b1_summary = patient_summary['hrsd_total_b1']
t30_summary = patient_summary['hrsd_total_t30']
assert(len(b1_summary) == len(t30_summary))
#calculate the %change of the depression outcome (keep it positive)
pChange_hrsd = abs((t30_summary - b1_summary)/b1_summary)*100
#start k_means
kmeans = KMeans(n_clusters=3, random_state=42)
data_1d = pChange_hrsd.to_frame()
labels = kmeans.fit_predict(X=data_1d)
#append to the df
patient_summary['pChanged'] = pChange_hrsd
patient_summary['cluster'] = labels
#save
patient_summary.to_excel('clustered_label.xlsx')

#plot the result
plt.figure(figsize=(10,3))
sns.stripplot(data=patient_summary, x='pChanged', hue='cluster', jitter=0.1, alpha=0.8, size=7)

plt.title('1D Clusters of Percent Change Outcomes')
plt.xlabel('Percent Change (%)')
plt.gca().yaxis.set_visible(False)
plt.show()


