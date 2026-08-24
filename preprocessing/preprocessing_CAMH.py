from glob import glob
import os
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import matplotlib.pyplot as plt
import numpy as np
from autoreject import AutoReject #Auto rejecting bad epoch
import pandas as pd
from collections import Counter #For counting repeating numbers/entries

"""
Preprocessing steps
GOAL:
Make it so that this preprocessing pipeline works for BIDS structures
"""

# path = f"/home/slee172/projects/def-sblain/slee/BIAPT_Lab/EEG_Preprocessing/EEG_dataset/CARTBIND_EEG/0_REST-EEG_data-CAMH"
# save_path = f"/home/slee172/projects/def-sblain/slee/BIAPT_Lab/CARTBIND/CARTBIND_CAMH"
path = '/Volumes/NINET/CARTBIND_EEG_data/0_REST-EEG_data-CAMH'
save_path = '/Volumes/NINET/CARTBIND_EEG_data/CARTBIND_CAMH'
mne.viz.set_browser_backend("matplotlib")

#Epoching
#Out of place operation, doesn't modify the input signal
#returns epoch
def generate_epoch(signal, report, duration=10):
    #Not done yet
    signal_epoch = mne.make_fixed_length_epochs(
        signal, 
        duration=duration, #10 second slice
        preload=True,
        reject_by_annotation=True)
    signal_epoch.pick_types(eeg=True)
    fig_epoch = signal_epoch.compute_psd(picks='eeg', fmax=50).plot(show=False)
    plt.title("PSD of Epochs")
    report.add_figure(
        fig=fig_epoch,
        title="Epochs PSD",
        image_format='PNG'
    )
    plt.close(fig_epoch)
    return signal_epoch

#Perform autoreject (Test diff param)
def reject_epoch(epochs, report, n_interpolate=[1, 4, 8, 12], consensus=[0.1, 0.2, 0.3, 0.4], cv=10): #Keep playing with these params
    picks = mne.pick_types(epochs.info, eeg=True, eog=False, exclude="bads")
    thresh_method = "bayesian_optimization"
    autoreject = AutoReject(n_interpolate=n_interpolate,
                            consensus=consensus,
                            cv=cv,
                            picks=picks,
                            thresh_method=thresh_method,
                            n_jobs=-1,
                            random_state=42,
                            verbose=False)
    ar_epoch, reject_log = autoreject.fit_transform(epochs, return_log=True)
    if any(reject_log.bad_epochs):
        fig_bad = epochs[reject_log.bad_epochs].plot(show=False, scalings=dict(eeg=100e-6))
        report.add_figure(
            fig=fig_bad, title='Autoreject bad epochs',
            image_format='PNG'
        )
        plt.close(fig_bad)
    if len(ar_epoch) != 0:
        fig_clean = ar_epoch.compute_psd(picks='eeg', fmax=50).plot(show=False)
        plt.title("PSD of epochs post ar")
        report.add_figure(
            fig=fig_clean,
            title="Post-Autoreject",
            image_format='PNG'
        )
        plt.close(fig_clean)
    #Rejection log
    bad_epochs_mask = reject_log.bad_epochs
    total_epochs = len(bad_epochs_mask)
    dropped_count = int(sum(bad_epochs_mask))
    dropped_pct = (dropped_count / total_epochs) * 100
    retained_count = total_epochs - dropped_count

    # Summary HTML
    summary_html = f"""
    <div style="font-family: sans-serif; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
        <h3 style="margin-top: 0;">Epoch Rejection Summary</h3>
        <table style="width: 100%; border-collapse: collapse; text-align: left;">
            <tr style="border-bottom: 1px solid #ddd;">
                <th style="padding: 8px;">Total Epochs Processed</th>
                <td style="padding: 8px;">{total_epochs}</td>
            </tr>
            <tr style="border-bottom: 1px solid #ddd;">
                <th style="padding: 8px;">Epochs Retained</th>
                <td style="padding: 8px;"><b>{retained_count}</b> ({100 - dropped_pct:.1f}%)</td>
            </tr>
            <tr style="border-bottom: 1px solid #ddd;">
                <th style="padding: 8px;">Epochs Dropped</th>
                <td style="padding: 8px; color: #d9534f;"><b>{dropped_count}</b> ({dropped_pct:.1f}%)</td>
            </tr>
        </table>
    </div>
    """

    report.add_html(
        html=summary_html,
        title="Epoch Summary",
        tags=("summary", "autoreject")
    )

    # Convert AutoReject Thresholds (Volts to µV) to DataFrame
    thresh_data = [
        {"Channel": ch, "Threshold (µV)": round(val * 1e6, 2)}
        for ch, val in autoreject.threshes_.items()
    ]
    df_thresh = pd.DataFrame(thresh_data)

    #Visualizing
    # 1. AutoReject Rejection Matrix Plot
    # Color-codes good, bad (dropped), and interpolated channels per epoch
    fig_reject = reject_log.plot(orientation='horizontal', show_names=1, show=False) #
    fig_reject.set_size_inches(8, 12)
    report.add_figure(
        fig=fig_reject,
        title="AutoReject Epoch Matrix (Bad & Interpolated Channels)",
        tags=("epochs", "autoreject", "visualization")
    )
    plt.close(fig_reject)

    # 2. Channel Threshold Barplot
    fig_thresh, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df_thresh["Channel"], df_thresh["Threshold (µV)"], color="#2b5c8f")
    ax.set_ylabel("Peak-to-Peak Threshold (µV)")
    ax.set_title("Peak-to-Peak Thresholds per Channel")
    ax.set_xticklabels(df_thresh["Channel"], rotation=90, fontsize=8)
    plt.tight_layout()

    report.add_figure(
        fig=fig_thresh,
        title="Channel Threshold Distribution",
        tags=("channels", "thresholds", "visualization")
    )
    plt.close(fig_thresh)

    # 3. Channel Threshold Table
    thresh_html = f"""
    <h3>Calculated Channel Thresholds</h3>
    {df_thresh.to_html(index=False, classes='table table-striped', justify='left')}
    """
    report.add_html(
        html=thresh_html,
        title="Threshold Data Table",
        tags=("channels", "thresholds", "table")
    )
    return ar_epoch

#h_trans_bandwidth=5
def filter_signal(raw, report, l_freq=0.5, h_freq=45.0, notch=60, downsample=250):
    # raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq).notch_filter(freqs=notch)
    #Try this one
    raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, h_trans_bandwidth=5.0).notch_filter(freqs=notch)
    filter_ds = raw_filtered.resample(downsample)

    #Autoreject bad channels
    autoreject_bads_ch(signal=filter_ds, report=report, zscore_thresh=2.0, samp_freq=250, n_fft=2048, fmax=45)
    # filter_ds.compute_psd(fmax=50, picks='eeg').plot()
    # plt.show(block=True)
    # #For manual bad channel selection
    # filter_ds.plot(block=True)
    report.add_raw(
        raw=filter_ds,
        title="Filter First & Downsample",
        psd=True
    )
    return filter_ds

#Mutates the signal
def set_reference(signal, report, ref='average'):
    misc_ch_names = [signal.ch_names[i] for i in mne.pick_types(signal.info, misc=True)]
    signal.drop_channels(misc_ch_names)
    # if misc_ch_names != []:
    #     signal.set_eeg_reference(ref_channels=misc_ch_names)
    #     signal.set_channel_types({misc_ch_names[0]: 'eeg'})
    # else:
    #     signal.add_reference_channels(ref_channels='CPz')
    #     signal.set_channel_types({'CPz': 'eeg'})
    signal.set_eeg_reference(ref_channels=ref)
    #Report reref
    report.add_raw(
        raw=signal,
        title=f"{ref} reference",
        psd=True
    )

#For EEG where HEOG and VEOG are present
#Divide by zero & matmul warnings!!! (Debug this)
def ica_remove_HVEOG(signal, report):
    signal.set_channel_types({'VEO': 'eog', 'HEO': 'eog'})
    eeg_picks = mne.pick_types(signal.info, eeg=True, meg=False, exclude='bads')
    n_eeg_channels = len(eeg_picks)
    ica = ICA(n_components=0.999999, 
                  random_state=97, 
                  max_iter="auto",
                  method='infomax', 
                  fit_params=dict(extended=True))
        
    ica_train = signal.copy()

    # # ===== Debug checks before ICA =====
    # print("ICA train bad ch: ", ica_train.info['bads'])
    # X = ica_train.get_data(picks='eeg')

    # print("EEG data shape:", X.shape)
    # print("Contains NaN:", np.isnan(X).any())
    # print("Contains Inf:", np.isinf(X).any())
    # print("Global max abs:", np.max(np.abs(X)))

    # print("\nEstimated rank:")
    # print(mne.compute_rank(ica_train))

    # print("\nPer-channel statistics:")
    # eeg_picks = mne.pick_types(
    #     ica_train.info,
    #     eeg=True,
    #     meg=False,
    #     exclude="bads"
    # )

    # for idx, pick in enumerate(eeg_picks):
    #     ch_name = ica_train.ch_names[pick]
    #     ch = X[idx]

    #     print(
    #         f"{ch_name:6s} "
    #         f"min={np.min(ch):.3e} "
    #         f"max={np.max(ch):.3e} "
    #         f"std={np.std(ch):.3e}"
    #     )

    #     if np.isnan(ch).any():
    #         print(f"  --> {ch_name} contains NaN")

    #     if np.isinf(ch).any():
    #         print(f"  --> {ch_name} contains Inf")
    # # ===== Fit ICA =====

    ica.fit(ica_train, picks='eeg')
    eog_indices, eog_scores = ica.find_bads_eog(ica_train, ch_name=['VEO', 'HEO'])
    ica.exclude = eog_indices
    cleaned_data = signal.copy()
    ica.apply(cleaned_data)

    #Report ICA
    report.add_ica(
        ica=ica,
        title="ICA EOG",
        inst=signal,
        eog_scores=eog_scores,
        n_jobs=None
    )
    fig = cleaned_data.compute_psd(picks='eeg',fmax=50).plot(show=False)
    plt.title("PSD after ocular artifact removed")
    report.add_figure(
        fig=fig,
        title="PSD post ICA",
        image_format='PNG'
    )
    plt.close(fig)
    return cleaned_data

#Use ICA_Label to remove artifacts
#Input signal here is filtered at 0.5~45hz
def remove_artifacts(signal, report):
    ica = ICA(n_components=None, 
              random_state=42, 
              max_iter="auto",
              method='infomax', 
              fit_params=dict(extended=True))
    
    ica_train = signal.copy().filter(l_freq=1.0, h_freq=45.0, h_trans_bandwidth=5.0).notch_filter(freqs=60.0)
    ica.fit(ica_train)
    labels = label_components(ica_train, ica, method='iclabel')
    
    target_labels = ['muscle', 'channel_noise', 'heart', 'line_noise']
    exclude_idx = [
        idx for idx, (label, prob) in enumerate(zip(labels['labels'], labels['y_pred_proba']))
        if label in target_labels and prob > 0.80
    ]
    
    ica.exclude = exclude_idx
    cleaned_data = signal.copy()
    ica.apply(cleaned_data)

    report.add_ica(
        ica=ica,
        title="ICA Artifact Removal (Filtered Thresholds)",
        inst=signal,
        n_jobs=None
    )
    
    label_data = [
        {"Component": f"ICA{idx:03d}", "Classification": lbl, "Confidence": f"{prob:.2%}"}
        for idx, (lbl, prob) in enumerate(zip(labels["labels"], labels["y_pred_proba"]))
    ]
    df_labels = pd.DataFrame(label_data)
    
    # Highlight the specific components that met the >80% probability threshold
    def highlight_bads(row):
        if row.name in exclude_idx:  
            return ['background-color: #ffcccc'] * len(row)
        return [''] * len(row)
        
    styled_df = df_labels.style.apply(highlight_bads, axis=1).hide(axis="index").to_html()

    labels_html = f"""
    <div style="font-family: sans-serif; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
        <h3 style="margin-top: 0;">ICLabel Component Classifications</h3>
        <p>Components classified as <b>muscle artifact, channel noise, heart beat,</b> or <b>line noise</b> with >80% probability have been automatically excluded.</p>
        {styled_df}
    </div>
    """
    
    report.add_html(
        html=labels_html,
        title="ICLabel Classifications",
        tags=("ica", "labels", "table")
    )

    #Spectral leakage???
    fig = cleaned_data.compute_psd(picks='eeg', fmax=50).plot(show=False)
    plt.title("PSD post ICA (Filtered)")
    report.add_figure(
        fig=fig,
        title="PSD post ICALabel",
        image_format='PNG'
    )
    plt.close(fig)
    
    return cleaned_data

#Mutates the raw object
#if DC correction is during the signal (start after the DC correction)
def crop_signal(raw, samp_freq=10000):
    samp_freq = samp_freq
    print(f"Raw information: {raw}")
    events, event_id = mne.events_from_annotations(raw)
    print(f"Events and corresponding id: {event_id}: {events}")
    """
    First row gives us the point where the recording started
    Second row gives us the position where the event occurs
    Third row gives us the position where the event ends
    """
    print(events.shape)
    event_start_val = event_id.get("Comment/Start Eyes Closed ")
    event_end_val = event_id.get("Comment/Stop Eyes Closed")
    event_dc_corr = event_id.get("DC Correction/")

    """Early exit"""
    if event_end_val is None:
        event_end_val = event_id.get("Comment/Start Eyes Open")
    
    if event_start_val:
        start_idx = events[:,2].tolist().index(event_start_val)
        event_start = events[start_idx, 0]
        print(f"Event begins: {event_start}")
        t_start = event_start/samp_freq
    else:
        t_start = 0.0 #use the default value

    if event_dc_corr:
        dc_corr_idx = events[:,2].tolist().index(event_dc_corr)
        event_dc = events[dc_corr_idx, 0]
        print(f"DC-correction happened at: {event_dc}")
        #comment this one out for now
        # t_start = event_dc/samp_freq if event_start_val is not None and (event_start < event_dc) else t_start 
            
    if event_end_val:
        end_idx = events[:,2].tolist().index(event_end_val)
        event_end = events[end_idx, 0]
        print(f"Event ends: {event_end}")
        t_end = event_end/samp_freq
    else:
        t_end = None #use the default value

    """Trim the eeg signal"""
    raw.crop(tmin=t_start, tmax=t_end)

"""Fix the mapping method here"""
#Mutates the raw object
def set_montage(raw, report, type='standard_1005'):
    montage = mne.channels.make_standard_montage(type)
    #To make this work for everything this needs to change
    rename_dict = {
        # Case mismatch corrections
        'FP1': 'Fp1', 'FPZ': 'Fpz', 'FP2': 'Fp2', 
        'FZ': 'Fz', 'FCZ': 'FCz', 'CZ': 'Cz', 
        'CPZ': 'CPz', 'PZ': 'Pz', 'POZ': 'POz', 'OZ': 'Oz',
        # Cerebellar aliases mapped to 10-05 positions (according to mne forum)
        #https://mne.discourse.group/t/raw-set-montage-cant-indentify-the-midline-electrodes/5455/2
        'CB1': 'POO7', 'CB2': 'POO8'
    }
    channel_types = {
        'VEO': 'eog', 
        'HEO': 'eog', 
        'EKG': 'ecg', 
        'EMG': 'emg',
        'M1' : 'misc',
        'M2' : 'misc'
    }
    raw.rename_channels(rename_dict)
    raw.set_channel_types(channel_types)
    raw.set_montage(montage=montage, on_missing='warn')
    fig = raw.plot_sensors(show_names=True)
    report.add_figure(
        fig=fig,
        title=f"{type} system plot",
        image_format='PNG'
    )
    plt.close(fig)

def save_result(obj, save_path, title, overwrite):
    if isinstance(obj, mne.Report):
        obj.save(fname=f"{save_path}/{title}", overwrite=overwrite)
    else:
        obj.save(f"{save_path}/{title}", overwrite=overwrite)

#Rejecting criteria
#Channels above certain standard deviation (2 or 3 std above)
#Higher limit of electrophysiological signal 
#50% of the time bad contact
"""
For the report, the following should be marked
1. What channels are marked bad + total #
2. percentage of region thrown out (ch in specific region. ex. frontal)
"""
def autoreject_bads_ch(signal, report, zscore_thresh, samp_freq, n_fft=2048, fmax=45.0):
    #mne picks
    picks = mne.pick_types(signal.info, eeg=True, exclude='bads')

    #computing psd
    spectrum = signal.compute_psd(fmax=fmax, picks='eeg', n_fft=n_fft)
    psd, freq = spectrum.get_data(return_freqs=True)
    ch_names = spectrum.ch_names
    
    #specific window of frequencies of interest
    # psd_scaled = np.array(psd*1e12)[:,24:]
    # freq = freq[24:]
    psd_scaled = np.array(psd*1e12)
    freq = freq

    mean_ch_power = np.mean(psd_scaled, axis=0) #scalar val per channel (only eeg)
    median_ch_power = np.median(psd_scaled, axis=0) #scalar val per channel (only eeg)

    #calculate std & zscore
    std_power = np.std(psd_scaled, axis=0)
    z_scores = np.abs(psd_scaled - mean_ch_power)/(std_power + 1e-12)
    bad_indices = np.where(z_scores > zscore_thresh)[0]

    #calculate the % of bad ch in the observed window
    length = len(freq) 
    counts = Counter(bad_indices)
    bads = [int(item) for item, count in counts.items() if count/length >= 0.50]
    bad_ch = [ch_names[bad] for bad in bads]

    #detect bad contact via peak-to-peak calculation
    data = signal.get_data(picks='eeg')
    # min_data = np.min(data, axis=1)
    # max_data = np.max(data, axis=1)

    n_ch, n_points = data.shape
    step = samp_freq*10 #10 second epoch
    total_step = n_points//step
    
    # bad_contact_dic = {i:0 for i in range(n_ch)}
    bad_contact_counts = np.zeros(n_ch, dtype=int)
    for i in range(0, n_points, step):
        epoch = data[:,i:i+step]
        #per epoch calculate min & max + index
        min_data = np.min(epoch, axis=1)
        max_data = np.max(epoch, axis=1)
        bad_lower_bound = np.where((max_data - min_data) <= 5e-6)[0] #np.where() returns a tuple (unpack the tuple)
        bad_upper_bound = np.where((max_data - min_data) >= 500e-6)[0]

        #Evaluates the quality of the channel (temporal)
        if bad_lower_bound is not []:
            bad_contact_counts[bad_lower_bound] += 1
        if bad_upper_bound is not []:
            bad_contact_counts[bad_upper_bound] += 1
        print('Upper bound: ', bad_upper_bound)
        print('Lower bound: ', bad_lower_bound)

        #Evaluates the quality of the channel for each epoch
        #Ex if for one epoch there are 50%> bad contacts mark it bad and drop it later
        #Should this be implemented or no????

    #calculate how bad
    print(bad_contact_counts)
    bad_contact_prob = bad_contact_counts/total_step
    bad_bounds_idx = np.where(bad_contact_prob >= 0.5)[0]
    bad_bound_ch = [ch_names[b] for b in bad_bounds_idx]


    """Segmented p2p calculation"""


    #lower bound
    # bad_lower_bound = np.where((max_data - min_data) <= 5e-6)[0] #np.where() returns a tuple (unpack the tuple)
    # bad_upper_bound = np.where((max_data - min_data) >= 500e-6)[0]
    # bad_bounds_idx = np.concatenate((bad_lower_bound, bad_upper_bound), axis=0)
    # bad_bound_ch = [ch_names[b] for b in bad_bounds_idx]
    

    bad_ch_names = list(set(bad_bound_ch + bad_ch))
    # bad_ch_names = bad_ch
    ratio = len(bad_ch_names)/len(ch_names)
    if ratio > 0.3:
        alert_html = f"""
        <div style="padding: 15px; margin: 10px 0; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;">
            <h3 style="margin-top:0;"> Skipping autoreject_bads_ch: Excessive Bad Channels</h3>
            <p><b>Bad Channel Ratio:</b> {ratio:.1%} ({len(bad_ch_names)} / {len(ch_names)} channels marked bad)</p>
            <p><b>Threshold:</b> 30.0%</p>
            <p><b>Bad Channels:</b> {', '.join(bad_ch_names)}</p>
        </div>
        """
        #report in mne.Report
        report.add_html(
            html=alert_html,
            title="Bad Channel Threshold Exceeded",
            section="Quality Control",
        )
    elif bad_ch_names:
        #mark bad
        signal.info['bads'] = list(set(signal.info["bads"] + bad_ch_names))


    #report everything in mne.Report
    fig, ax = plt.subplots(figsize=(8, 4))
    # Plot normal channels in gray, flagged bad channels in orange/red
    print(ch_names)
    print("Bad ch: ", signal.info['bads'])
    print(len(ch_names))
    print("Shape of the psd: ", psd_scaled.shape)
    for i, ch_name in enumerate(ch_names):
        if ch_name in bad_ch_names:
            ax.plot(
                freq,
                psd_scaled[i],
                color="orange",
                alpha=0.6,
                linewidth=1,
                label="Bad Channel" if i == 0 else "",
            )
        else:
            ax.plot(freq, psd_scaled[i], color="gray", alpha=0.2, linewidth=0.8)

    # Plot Global Reference Baselines
    ax.plot(
        freq,
        mean_ch_power,
        color="red",
        linewidth=2,
        label="Global Mean PSD",
    )
    ax.plot(
        freq,
        median_ch_power,
        color="blue",
        linewidth=2,
        linestyle="--",
        label="Global Median PSD",
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title(f"PSD Spectrum Baseline (Flagged {len(bad_ch_names)} channels)")
    ax.grid(True)

    # Clean up duplicate legend labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    # Pass plot directly into MNE Report
    if report is not None:
        report.add_figure(
            fig=fig,
            title="Auto-Reject Bad Channels (PSD Analysis)",
            caption=f"Rejected Channels: {', '.join(bad_ch_names) if bad_ch_names else 'None'}",
            section="Quality Control",
        )
    plt.close(fig)


def main():
    dropped = []
    folders = glob(f"{path}/*")
    epoch_num = []
    for folder in sorted(folders):
        ID = os.path.basename(folder).split("CBN")[1]
        print(ID)
        subfolders = sorted(glob(f"{folder}/*"))
        #Subfolders ex. T0, T1, T2
        print(subfolders)
        for file in subfolders:
            #If not eyesclosed recording skip
            if os.path.basename(file).split("_")[-1] != "EYESCLOSED.cnt":
                continue

            time = os.path.basename(file).split("_")[1]
            #MNE Report
            report_title = f"CAMH_CARTBIND_{ID}_{time}"
            report = mne.Report(title=report_title)

            if file == []:
                print(f"=======File is empty: Skipping {ID}=======")
                continue
            else:
                print(f"=======Processing {ID} Time {time}=======")
            """
            Filter logic +
            Artifact + other methods that change the raw signal
            should come here
            """    
            raw = mne.io.read_raw_cnt(file, preload=False, data_format='int32')
            # crop_signal(raw=raw)
            raw.load_data()
            set_montage(raw=raw, report=report, type='standard_1005')
            filtered_signal = filter_signal(raw=raw, report=report)
            cleaned_signal = ica_remove_HVEOG(signal=filtered_signal, report=report)
            if cleaned_signal == None:
                dropped.append(f"{ID}_{time}")
                continue
            #Drop all bad channels (not needed for feature generations)
            cleaned_signal.drop_channels(raw.info['bads'])
            save_result(obj=cleaned_signal, save_path=f"{save_path}/{time}_filtered_eeg", 
                        title=f"CBN02_{ID}_REST_EC_{time}_CAMH_filtered_eeg.fif", overwrite=True)
            
            set_reference(signal=cleaned_signal, report=report)
            epochs = generate_epoch(signal=cleaned_signal, report=report)
            ar_epochs = reject_epoch(epochs=epochs, report=report)
            #Save epoch
            save_result(obj=ar_epochs, save_path=f"{save_path}/{time}_epoch", 
                        title=f"CBN02_{ID}_REST_EC_{time}_CAMH_epo.fif", overwrite=True)
            #Save mne Report
            save_result(obj=report, save_path=f"{save_path}/Reports/{time}", 
                        title=f"{report_title}_report.html", overwrite=True)

if __name__ =="__main__":
    main()
    #counting the overlap
    # folders = glob(f"{path}/*")
    # epoch_num = []
    # overlap = []
    # for folder in sorted(folders):
    #     ID = os.path.basename(folder).split("CBN")[1]
    #     print(ID)
    #     subfolders = sorted(glob(f"{folder}/*"))
    #     #Subfolders ex. T0, T1, T2
    #     rest_EC = []
    #     for file in subfolders:
    #         if "EYESCLOSED" not in os.path.basename(file).upper():
    #             continue
    #         else:
    #             rest_EC.append(file)
    #     if len(rest_EC) >= 3:
    #         overlap.append(ID)
    # print(overlap)
    # print("Number of overlaps: ", {len(overlap)})

