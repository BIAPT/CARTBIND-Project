import ordpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import seaborn as sns
import os
import mne

time = 'T2'
dir_path = f'/Volumes/NINET/CARTBIND_EEG_data/CARTBIND_CAMH/{time}_epoch'
save_path = f'/Volumes/NINET/CARTBIND_EEG_data/CARTBIND_CAMH/Features/{time}_dataset'

feature_path = f'/Volumes/NINET/CARTBIND_EEG_data/CARTBIND_CAMH/Features'

rem_idx = [4, 5, 10, 12, 13, 15, 19, 23, 31, 66]
resp_idx = [22, 26, 33, 34, 41, 43, 47, 51, 54, 58, 60, 67]
non_resp_idx = [0, 1, 2, 3, 6, 7, 8, 9, 11, 14, 16, 17, 18, 20, 21, 24, 25, 27, 28, 29, 30, 32, 
                35, 36, 37, 38, 39, 40, 42, 44, 45, 46, 48, 49, 50, 52, 53, 55, 56, 57, 59, 61, 
                62, 63, 64, 65, 68, 69, 70]

#computes PEn & Stat. comp for given Epoch (mne)
def calc_comp_ent(signal, dim, delay):
    h,c = ordpy.complexity_entropy(data=signal, dx=dim, dy=1, taux=delay, tauy=1)
    return h,c

#calculates the ideal tau
def compute_ami(signal, max_lag=60, bins=32):
    """
    Computes Average Mutual Information (AMI) for delays up to max_lag.
    """
    N = len(signal)
    ami_values = []

    for tau in range(1, max_lag + 1):
        # Create delayed vectors
        x = signal[:-tau] #from (start to end - tau)
        y = signal[tau:] #from (tau to end)
        
        # 2D Histogram to estimate joint probability density P(x, y)
        pxy, x_edges, y_edges = np.histogram2d(x, y, bins=bins, density=True)
        
        # Convert density to probability distributions
        pxy = pxy / np.sum(pxy)
        px = np.sum(pxy, axis=1)  # Marginal distribution P(x)
        py = np.sum(pxy, axis=0)  # Marginal distribution P(y)
        
        # Compute Mutual Information: sum( P(x,y) * log2( P(x,y) / (P(x)*P(y)) ) )
        # Mask zero probabilities to prevent log(0) errors
        nz = pxy > 0
        px_py = np.outer(px, py)
        mi = np.sum(pxy[nz] * np.log2(pxy[nz] / px_py[nz]))
        
        ami_values.append(mi)
        
    return np.array(ami_values)

#plots the complexity-entropy plane
def plot_CECP(points: tuple, title, dx):
    #extract the corresponding points
    x_t0, y_t0 = points[0]
    x_t1, y_t1 = points[1]
    x_t2, y_t2 = points[2]

    #calculate min/max comp&ent
    min_comp = ordpy.minimum_complexity_entropy(dx=dx)
    max_comp = ordpy.maximum_complexity_entropy(dx=dx)
    H_min = min_comp[:,0]
    H_max = max_comp[:,0]
    C_min = min_comp[:,1]
    C_max = max_comp[:,1]

    #synthetic signal white noise (Gaussian)
    n_sample = 10000
    mean = 0.0
    std = 1.0
    noise_gauss = np.random.normal(mean, std, n_sample)
    H_wn, C_wn = ordpy.complexity_entropy(data=noise_gauss, dx=dx)

    #plotting
    plt.figure(figsize=(7, 4.5))


    """coloring the chaos region"""
    #write here

    """indicate the stochastic region"""
    #here

    """draw the transition line"""



    plt.plot(H_min, C_min, 'k--', label='$C_{\mathrm{min}}$')
    plt.plot(H_max, C_max, 'k-', label='$C_{\mathrm{max}}$')

    #plotting the points
    plt.scatter(x_t0, y_t0, alpha=0.7, label='T0 rem', color="#ff7f0e")
    plt.scatter(x_t1, y_t1, alpha=0.7, label='T0 resp', color="#6eff0e")
    plt.scatter(x_t2, y_t2, alpha=0.7, label='T0 non-resp', color="#0e8fff")

    #plot the white noise
    plt.scatter(
        H_wn,
        C_wn,
        color='red', 
        marker='*',       
        s=150,             
        zorder=10,         
        label='White Noise'
    )

    plt.xlabel('Permutation Entropy ($H$)')
    plt.ylabel('Statistical Complexity ($C$)')
    plt.title(f"{title} dx={dx}")
    plt.xlim(0, 1)
    plt.ylim(0, max(C_max) * 1.05)
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.show()

#Maybe try dim=6??????


def extract_data(data: mne.Epochs, ch):
    #find the overlap channels
    overlap = list(set(data.ch_names) & set(ch))
    return data.get_data(picks=overlap)

def plot_average_direction():
    #calculate min/max comp&ent
    min_comp = ordpy.minimum_complexity_entropy(dx=5)
    max_comp = ordpy.maximum_complexity_entropy(dx=5)
    H_min = min_comp[:,0]
    H_max = max_comp[:,0]
    C_min = min_comp[:,1]
    C_max = max_comp[:,1]

    #synthetic signal white noise (Gaussian)
    n_sample = 10000
    mean = 0.0
    std = 1.0
    noise_gauss = np.random.normal(mean, std, n_sample)
    H_wn, C_wn = ordpy.complexity_entropy(data=noise_gauss, dx=5)

    #plotting
    plt.figure(figsize=(7, 4.5))


    """coloring the chaos region"""
    #write here

    """indicate the stochastic region"""
    #here

    """draw the transition line"""



    plt.plot(H_min, C_min, 'k--', label='$C_{\mathrm{min}}$')
    plt.plot(H_max, C_max, 'k-', label='$C_{\mathrm{max}}$')

    #plot the white noise
    plt.scatter(
        H_wn,
        C_wn,
        color='red', 
        marker='*',       
        s=150,             
        zorder=10,         
        label='White Noise'
    )

    plt.xlabel('Permutation Entropy ($H$)')
    plt.ylabel('Statistical Complexity ($C$)')

    #load data
    t0 = np.load(file=f'{feature_path}/T0_dataset/frontal_dim_5_delay_9_CAMH_T0.npy')
    t1 = np.load(file=f'{feature_path}/T1_dataset/frontal_dim_5_delay_9_CAMH_T1.npy')
    t2 = np.load(file=f'{feature_path}/T2_dataset/frontal_dim_5_delay_9_CAMH_T2.npy')

    #get points
    resp_T0 = np.array([row for idx, row in enumerate(t0) if idx in resp_idx])
    resp_T1 = np.array([row for idx, row in enumerate(t1) if idx in resp_idx])
    resp_T2 = np.array([row for idx, row in enumerate(t2) if idx in resp_idx])

    resp_x_T0 = resp_T0[:,0]
    resp_y_T0 = resp_T0[:,1]
    resp_x_T1 = resp_T1[:,0]
    resp_y_T1 = resp_T1[:,1]
    resp_x_T2 = resp_T2[:,0]
    resp_y_T2 = resp_T2[:,1]


    color =["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # plt.figure('Average direction')
    #resp
    plt.figure('resp')
    x_T0 = np.mean(resp_x_T0,axis=0)
    y_T0 = np.mean(resp_y_T0, axis=0)

    x_T1 =np.mean(resp_x_T1,axis=0)
    y_T1 = np.mean(resp_y_T1, axis=0)

    x_T2 = np.mean(resp_x_T2,axis=0)
    y_T2=np.mean(resp_y_T2,axis=0)

    plt.scatter(x_T0, y_T0, label='T0', marker='o', facecolors='none', edgecolors=color[3])
    plt.scatter(x_T1, y_T1, label='T1', marker='^', color=color[2])
    plt.scatter(x_T2, y_T2, label='T2', marker='o', color=color[0])

    plt.annotate('', 
                    xy=(x_T1, y_T1),      # Arrow head target
                    xytext=(x_T0, y_T0),  # Arrow tail start
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, alpha=0.6,
                    shrinkA=0,
                    shrinkB=0))
        
        # Arrow from T1 -> T2
    plt.annotate('', 
                    xy=(x_T2, y_T2),      # Arrow head target
                    xytext=(x_T1, y_T1),  # Arrow tail start
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5, alpha=0.6,
                    shrinkA=0,
                    shrinkB=0))
    plt.xlabel("PEn")
    plt.ylabel("LMC (Statistical complexity")


    #get points
    non_resp_T0 = np.array([row for idx, row in enumerate(t0) if idx in non_resp_idx])
    non_resp_T1 = np.array([row for idx, row in enumerate(t1) if idx in non_resp_idx])
    non_resp_T2 = np.array([row for idx, row in enumerate(t2) if idx in non_resp_idx])

    non_resp_x_T0 = non_resp_T0[:,0]
    non_resp_y_T0 = non_resp_T0[:,1]
    non_resp_x_T1 = non_resp_T1[:,0]
    non_resp_y_T1 = non_resp_T1[:,1]
    non_resp_x_T2 = non_resp_T2[:,0]
    non_resp_y_T2 = non_resp_T2[:,1]


    #non-resp
    plt.figure('Non-Responder Plot')
    x_T0_2 = np.mean(non_resp_x_T0,axis=0)
    y_T0_2 = np.mean(non_resp_y_T0, axis=0)

    x_T1_2 =np.mean(non_resp_x_T1,axis=0)
    y_T1_2 = np.mean(non_resp_y_T1, axis=0)

    x_T2_2 = np.mean(non_resp_x_T2,axis=0)
    y_T2_2 = np.mean(non_resp_y_T2,axis=0)

    plt.scatter(x_T0_2, y_T0_2, label='T0', marker='o', facecolors='none', edgecolors=color[3])
    plt.scatter(x_T1_2, y_T1_2, label='T1', marker='^', color=color[2])
    plt.scatter(x_T2_2, y_T2_2, label='T2', marker='o', color=color[0])

    plt.annotate('', 
                    xy=(x_T1_2, y_T1_2),      # Arrow head target
                    xytext=(x_T0_2, y_T0_2),  # Arrow tail start
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, alpha=0.6,
                    shrinkA=0,
                    shrinkB=0))
        
        # Arrow from T1 -> T2
    plt.annotate('', 
                    xy=(x_T2_2, y_T2_2),      # Arrow head target
                    xytext=(x_T1_2, y_T1_2),  # Arrow tail start
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5, alpha=0.6,
                    shrinkA=0,
                    shrinkB=0))
    plt.xlabel("PEn")
    plt.ylabel("LMC (Statistical complexity")

    rem_T0 = np.array([row for idx, row in enumerate(t0) if idx in rem_idx])
    rem_T1 = np.array([row for idx, row in enumerate(t1) if idx in rem_idx])
    rem_T2 = np.array([row for idx, row in enumerate(t2) if idx in rem_idx])

    rem_x_T0 = rem_T0[:,0]
    rem_y_T0 = rem_T0[:,1]
    rem_x_T1 = rem_T1[:,0]
    rem_y_T1 = rem_T1[:,1]
    rem_x_T2 = rem_T2[:,0]
    rem_y_T2 = rem_T2[:,1]

    #rem
    plt.figure('Remission Plot')
    x_rem_T0 = np.mean(rem_x_T0,axis=0)
    y_rem_T0 = np.mean(rem_y_T0,axis=0)

    x_rem_T1 = np.mean(rem_x_T1,axis=0)
    y_rem_T1 = np.mean(rem_y_T1,axis=0)

    x_rem_T2 = np.mean(rem_x_T2,axis=0)
    y_rem_T2 = np.mean(rem_y_T2,axis=0)

    plt.scatter(x_rem_T0, y_rem_T0, label='T0', marker='o', facecolors='none', edgecolors=color[3])
    plt.scatter(x_rem_T1, y_rem_T1, label='T1', marker='^', color=color[2])
    plt.scatter(x_rem_T2, y_rem_T2, label='T2', marker='o', color=color[0])

    plt.annotate('', 
                    xy=(x_rem_T1, y_rem_T1),      # Arrow head target
                    xytext=(x_rem_T0, y_rem_T0),  # Arrow tail start
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, alpha=0.6,
                    shrinkA=0,
                    shrinkB=0))
        
        # Arrow from T1 -> T2
    plt.annotate('', 
                    xy=(x_rem_T2, y_rem_T2),      # Arrow head target
                    xytext=(x_rem_T1, y_rem_T1),  # Arrow tail start
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5, alpha=0.6,
                    shrinkA=0,
                    shrinkB=0))
    plt.xlabel("PEn")
    plt.ylabel("LMC (Statistical complexity")
    plt.title(f"Average Trend in T0->T1->T2 (Temporal)")
    plt.show()

#For starting point
def plot_violine(points: tuple):
    df_0 = pd.DataFrame(points[0])
    df_1 = pd.DataFrame(points[1])
    df_2 = pd.DataFrame(points[2])
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), sharex=True, sharey=True)

    # 3. Plot the first chart (Index 0)
    sns.kdeplot(
        data=df_0,
        x="H",
        y="C",
        cmap="Blues",
        fill=True,
        thresh=0.05,
        ax=axes[0],
    )
    axes[0].scatter(
        df_0["H"],  # Pass X data directly without 'x='
        df_0["C"],  # Pass Y data directly without 'y='
        color="black",
        s=5,
        alpha=0.3,
        label="Points",
        zorder=2,  # Forces the points to sit on top layer
    )
    axes[0].set_title("Remission Group (T0)")

    # 4. Plot the second chart (Index 1)
    sns.kdeplot(
        data=df_1,
        x="H",
        y="C",
        cmap="Oranges",
        fill=True,
        thresh=0.05,
        ax=axes[1],
    )
    axes[1].scatter(
        df_1["H"],  # Pass X data directly without 'x='
        df_1["C"],  # Pass Y data directly without 'y='
        color="black",
        s=5,
        alpha=0.3,
        label="Points",
        zorder=2,  # Forces the points to sit on top layer
    )
    axes[1].set_title("Resp Group (T0)")

    # 5. Plot the third chart (Index 2)
    sns.kdeplot(
        data=df_2,
        x="H",
        y="C",
        cmap="Greens",
        fill=True,
        thresh=0.05,
        ax=axes[2],
    )
    axes[2].scatter(
        df_2["H"],  # Pass X data directly without 'x='
        df_2["C"],  # Pass Y data directly without 'y='
        color="black",
        s=5,
        alpha=0.3,
        label="Points",
        zorder=2,  # Forces the points to sit on top layer
    )
    axes[2].set_title("Non-resp Group (T0)")

    # 6. Clean up layout spacing and render
    plt.tight_layout()
    plt.show()


def L2_norm_violine(points: tuple):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), sharex=True, sharey=True)
    sns.violinplot(data=points[0], y='L2', color="#aee297", ax=axes[0]) 
    axes[0].set_title("Non-resp Group")

    sns.violinplot(data=points[1], y='L2', color='#fab1a0', ax=axes[1])
    axes[1].set_title("Resp Group")

    sns.violinplot(data=points[2], y='L2', color='#74b9ff', ax=axes[2]) 
    axes[2].set_title("Rem Group")  

    plt.tight_layout()
    plt.show()


def plot_idiv_dir(points: tuple):

    x_T0, y_T0 = points[0]
    x_T1, y_T1 = points[1]
    x_T2, y_T2 = points[2]

    color =["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    plt.scatter(x_T0, y_T0, label='T0', marker='o', facecolors='none', edgecolors=color[3])
    plt.scatter(x_T1, y_T1, label='T1', marker='^', color=color[2])
    plt.scatter(x_T2, y_T2, label='T2', marker='o', color=color[0])

    plt.annotate('', 
        xy=(x_T1, y_T1),      # Arrow head target
        xytext=(x_T0, y_T0),  # Arrow tail start
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, alpha=0.6,
        shrinkA=0,
        shrinkB=0))
        
        # Arrow from T1 -> T2
    plt.annotate('', 
        xy=(x_T2, y_T2),      # Arrow head target
        xytext=(x_T1, y_T1),  # Arrow tail start
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5, alpha=0.6,
        shrinkA=0,
        shrinkB=0))
    plt.xlabel("PEn")
    plt.ylabel("LMC (Statistical complexity")
    plt.show(block=True)


def stacked_bar_plot(data):
    df = pd.DataFrame(data=data)
    df.set_index('group', inplace=True)
    df.plot(kind='bar', stacked=True, figsize=(8,6))

    plt.title('Per Group Trajectory')
    plt.ylabel('Percent (%)')
    plt.xlabel('Group')
    plt.xticks(rotation=0) # Keep x-axis labels upright
    plt.legend(title='Categories')

    # 4. Display the chart
    plt.show()
    

def main():
    frontal_channels = [
        'Fp1', 'Fpz', 'Fp2', 
        'AF3', 'AF4', 
        'F7', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 
        'FT7', 
        'FC5', 'FC3', 'FCz', 'FC2', 'FC4'
    ]
    temporal_core = ["FT7", "FT8", "T7", "T8", "TP7", "TP8"]
    h_list = []
    c_list = []
    delays = []
    ami_tensor = []
    for files in sorted(os.listdir(path=dir_path)):
        file_name = os.path.join(dir_path, files)
        #load epoch
        epoch = mne.read_epochs(fname=file_name,preload=True)
        #extrach important chanells
        frontal_values = extract_data(data=epoch, ch=frontal_channels)
        # h,c = calc_comp_ent(signal=np.mean(frontal_values, axis=(0,1)), dim=5, delay=9)
        # h_list.append(h)
        # c_list.append(c)
        # Average across epoch to stabilize the distribution
        result = compute_ami(np.mean(frontal_values, axis=(0,1)))

        peak_val, _ = find_peaks(x=-result)
        delays.append(peak_val[0])
    print(f"Delays: {delays}\nAverage Delay: {np.mean(delays)}")
    #plot the distribution of the delay
    sns.kdeplot(data=delays, fill=True, color='skyblue', alpha=0.4)
    plt.axvline(x=np.mean(delays), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(delays):.2f}")
    plt.title('Distribution of the delay')
    plt.xlabel('Delay (tau)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # combined = np.column_stack(tup=(h_list, c_list))
    # np.save(file=f"{save_path}/frontal_dim_5_delay_9_CAMH_{time}.npy", arr=combined)

def plot():
    t0_points = np.load(file=f'{feature_path}/T0_dataset/dim_5_delay_9_CAMH_T0.npy')
    t1_points = np.load(file=f'{feature_path}/T1_dataset/dim_5_delay_9_CAMH_T1.npy')
    t2_points = np.load(file=f'{feature_path}/T2_dataset/dim_5_delay_9_CAMH_T2.npy')

    t0_points_2 = np.array([row for idx, row in enumerate(t0_points) if idx in rem_idx])
    t1_points_2 = np.array([row for idx, row in enumerate(t0_points) if idx in resp_idx])
    t2_points_2 = np.array([row for idx, row in enumerate(t0_points) if idx in non_resp_idx])


    t0_resp = np.array([row for idx, row in enumerate(t0_points) if idx in resp_idx])
    t1_resp = np.array([row for idx, row in enumerate(t1_points) if idx in resp_idx])
    t2_resp = np.array([row for idx, row in enumerate(t2_points) if idx in resp_idx])


    t0_non_resp = np.array([row for idx, row in enumerate(t0_points) if idx in non_resp_idx])
    t1_non_resp = np.array([row for idx, row in enumerate(t1_points) if idx in non_resp_idx])
    t2_non_resp = np.array([row for idx, row in enumerate(t2_points) if idx in non_resp_idx])
    

    t0_rem = np.array([row for idx, row in enumerate(t0_points) if idx in rem_idx])
    t1_rem = np.array([row for idx, row in enumerate(t1_points) if idx in rem_idx])
    t2_rem = np.array([row for idx, row in enumerate(t2_points) if idx in rem_idx])


    t0 = (t0_points_2[:,0], t0_points_2[:,1])
    t1 = (t1_points_2[:,0], t1_points_2[:,1])
    t2 = (t2_points_2[:,0], t2_points_2[:,1])

    # t0 = (non_resp_diff[:,0], non_resp_diff[:,1])
    # t1 = (resp_diff[:,0], resp_diff[:,1])
    # t2 = (rem_diff[:,0], rem_diff[:,1])
    # plt.scatter(x=non_resp_diff[:,0]**2, y=non_resp_diff[:,1]**2, color='red')
    # plt.scatter(x=resp_diff[:,0]**2, y=resp_diff[:,1]**2, color='blue')
    # plt.scatter(x=rem_diff[:,0]**2, y=rem_diff[:,1]**2, color='green')

    #calculate the diff L2 norm
    




    # plt.show()
    
    
    #plot_CECP(points=(t0, t1, t2), title='Remission plot', dx=5)
    """For plot_violine"""
    # cecp_data_0 = {
    #         'H': t0[0],
    #         'C': t0[1]
    # }
    # cecp_data_1 = {
    #             'H': t1[0],
    #             'C': t1[1]
    # }
    # cecp_data_2 = {
    #             'H': t2[0],
    #             'C': t2[1]
    # }
    # plot_violine(points=(cecp_data_0, cecp_data_1, cecp_data_2))

    """For L2_norm_violine"""
    # resp_diff = np.sqrt(np.sum((t2_resp - t0_resp)**2, axis=1))
    # non_resp_diff = np.sqrt(np.sum((t2_non_resp - t0_non_resp)**2, axis=1))
    # rem_diff = np.sqrt(np.sum((t2_rem - t0_rem)**2, axis=1))
    # data_0 = pd.DataFrame(data={'L2': non_resp_diff})
    # data_1 = pd.DataFrame(data={'L2': resp_diff})
    # data_2 = pd.DataFrame(data={'L2': rem_diff})
    # L2_norm_violine(points=(data_0, data_1, data_2))

    """For plot_indiv_dir"""
    # for t0, t1, t2 in zip(t0_non_resp, t1_non_resp, t2_non_resp):
    #     plot_idiv_dir(points=(t0, t1, t2))

    """For stacked_bar_plot"""
    data = {
        'group': ['non-resp', 'resp', 'rem'],
        '↑↑': [14.3,16.7,30],
        '↓↓': [24.5,25,10],
        '↑↓': [36.7,16.7,20],
        '↓↑': [24.5,41.6,40],
    }
    stacked_bar_plot(data=data)




if __name__ == '__main__':
    # main()
    # plot()
    plot_average_direction()
