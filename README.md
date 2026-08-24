# CARTBIND Analysis
This repository contains different analysis on the A Canadian rTMS treatment and biomarker network in depression (CARTBIND) dataset.  
Analysis mainly involves observing the trajectory of different state (baseline, first TMS, final TMS) in the Complexity & Entropy Causal Plane (CECP). 
Specifically, calculating the permutation entropy (x-axis) and statistical complexity (y-axis) values per participants in each state & plot in the CECP.  
This repository contains four main directories.  
## preprocessing  
This folder contains preprocessing scrips for eeg recordings that took place at UBC, CAMH, UHN.  
Additionally, the run_preprocessing.sh is a bash script used for submitting job(s) to the Digital Research Alliance of Canada, a high computing cluster.  
For running the code in cluster, make sure to use 'sbatch' command and the name of the bash script.  
```bash
sbatch run_preprocessing.sh
```
### The preprocessing steps are as follows  
* **Crop Signal**  
For signals with mne annotations, the method "crop_signal" correctly crops the signal as per annotation.   
Make sure to edit the crop_signal method as the strings used for the annotation is hard coded in the current version.  
* **Set Montage**  
For UBC location, use  
&emsp;* standard_1020  
For CAMH & UHN  
&emsp;* standard_1005  
* **Filtering**  
This method performs zero-phase FIR Filter default (0.5~45hz) with notch filter at 60hz.  
For details please check the filter_signal() method
For detecting the bad signal/channel, after the filtering, this method internally calls 'autoreject_bads_ch()'.   
* **Autoreject bad channels**  
It looks at the standard deviation of the psd of all channels from the mean psd and if it is zscore_threshold away from the mean,  
it is marked as bad automatically. In addition, the method also calculates the peak-to-peak value of all the channels (lower bound: 5e-6, upper bound: 500e-6). Instead of using mne.Epoch, for p2p calculations, the data is loaded directly from the data (raw object) in ndarray format.  
Then we create the window/epoch-like segments (10 seconds) and if the channel is bad 50%>= it is marked bad & dropped later in the preprocessing pipeline.  
* **Artifact Removal**  
As the name suggests, the ica_remove_HVEOG() method will use the HEOG, VEOG recording to remove ocular artifacts from the EEG signals.  
Don't confuse ica_remove_HVEOG() with remove_artifacts(). This method doesn't require additional signals as it uses mne-ICALabel (pre-trained model for detecting artifacts) on the EEG signals and removes any artifacts that are above the threshold (chance of being artifact > 80%). 
This CARTBIND analysis does not use remove_artifacts() method.  
* **Rereferencing**  
Re-reference to 'average' referencing by default.    
* **Epoching**  
By default, generates 10s epoch and applies autoreject to remove any bad epoch.  
Finally, mne.Report is generated to verify every preprocessing steps and check the quality of the signals before/after at any steps.  
## features  
This folder contains code for generating features from filtered & epoched data saved after running the preprocessing script.  
feature_gen_auto.py file is a code that works with Derek's old PICU repository for running the feature extraction code sequentially in the cluster.  
## dataset  
This folder contains code that converts features to structured format (ex. np.array or pandas) that can be passed to different visualization scripts to generate figures.  
## visuals  
This folder contains scripts used to generate figures/plots for the CECP analysis (trajectory + initial points), direction & magnitude of pre post treatment, and more.  Many of the visual scripts are still work in progress, so readability is certainly limited.  

## Reminder  
+ Add env & conda env  
+ Add other scripts used in the cluster  