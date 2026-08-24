import matplotlib.pyplot as plt
import numpy as np
import ordpy #packages for comp & ent
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset #for zooming in plt plots

name = 'CAMH'
group = 'tri'

region = 'frontal'

#Extrac the dataset
feature_path = f"/Volumes/NINET/CARTBIND_EEG_data/CARTBIND_{name}/Features"
# T0_feature = np.load(f"{feature_path}/T0_dataset/T0_feature_collection_{region}_CARTBIND.npy")
# T1_feature = np.load(f"{feature_path}/T1_dataset/T1_feature_collection_{region}_CARTBIND.npy")
# T2_feature = np.load(f"{feature_path}/T2_dataset/T2_feature_collection_{region}_CARTBIND.npy")

T0_feature = np.load(file=f'{feature_path}/T0_dataset/dim_5_delay_9_CAMH_T0.npy')
T1_feature = np.load(file=f'{feature_path}/T1_dataset/dim_5_delay_9_CAMH_T1.npy')
T2_feature = np.load(file=f'{feature_path}/T2_dataset/dim_5_delay_9_CAMH_T2.npy')

subject_ids = [
    '003', '004', '005', '006', '008', '010', '011', '012', '013', '015',
    '018', '019', '021', '023', '024', '026', '029', '030', '031', '032',
    '034', '035', '036', '041', '043', '044', '045', '048', '051', '052',
    '054', '056', '058', '059', '061', '062', '063', '064', '067', '068',
    '069', '072', '073', '074', '075', '077', '078', '082', '083', '087',
    '088', '089', '091', '093', '094', '095', '096', '097', '098', '099',
    '102', '106', '112', '119', '120', '122', '123', '125', '127', '130', '136'
]

#resp indices
if group == 'bi':
    if name == 'UBC':
        resp_idx = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 19, 20, 22, 23, 24, 25, 26, 29, 31, 32, 34, 35, 37]
        non_resp_idx = [0, 4, 11, 14, 15, 16, 17, 18, 21, 27, 28, 30, 33, 36, 38, 39, 40]
    else:
        resp_idx = [4, 5, 10, 12, 13, 15, 19, 22, 23, 26, 31, 33, 34, 41, 43, 47, 51, 54, 58, 60, 66, 67]

        non_resp_idx = [0, 1, 2, 3, 6, 7, 8, 9, 11, 14, 16, 17, 18, 20, 21, 24, 25, 27, 28, 29, 30, 32, 
                        35, 36, 37, 38, 39, 40, 42, 44, 45, 46, 48, 49, 50, 52, 53, 55, 56, 57, 59, 61, 
                        62, 63, 64, 65, 68, 69, 70]

if group == 'tri':
    if name == 'CAMH':
        rem_idx = [4, 5, 10, 12, 13, 15, 19, 23, 31, 66]
        resp_idx = [22, 26, 33, 34, 41, 43, 47, 51, 54, 58, 60, 67]
        non_resp_idx = [0, 1, 2, 3, 6, 7, 8, 9, 11, 14, 16, 17, 18, 20, 21, 24, 25, 27, 28, 29, 30, 32, 
                        35, 36, 37, 38, 39, 40, 42, 44, 45, 46, 48, 49, 50, 52, 53, 55, 56, 57, 59, 61, 
                        62, 63, 64, 65, 68, 69, 70]

    
#data for T0/T1/T2
x_coord_T0 = T0_feature[:,0]
y_coord_T0 = T0_feature[:,1]

x_coord_T1 = T1_feature[:,0]
y_coord_T1 = T1_feature[:,1]

x_coord_T2 = T2_feature[:,0]
y_coord_T2 = T2_feature[:,1]

#resp/non_resp data
if group == 'bi':
    non_resp_x_T0 = [row for i, row in enumerate(x_coord_T0) if i in non_resp_idx]
    non_resp_y_T0 = [row for i, row in enumerate(y_coord_T0) if i in non_resp_idx]

    non_resp_x_T1 = [row for i, row in enumerate(x_coord_T1) if i in non_resp_idx]
    non_resp_y_T1 = [row for i, row in enumerate(y_coord_T1) if i in non_resp_idx]

    non_resp_x_T2 = [row for i, row in enumerate(x_coord_T2) if i in non_resp_idx]
    non_resp_y_T2 = [row for i, row in enumerate(y_coord_T2) if i in non_resp_idx]

    resp_x_T0 = [row for i, row in enumerate(x_coord_T0) if i in resp_idx]
    resp_y_T0 = [row for i, row in enumerate(y_coord_T0) if i in resp_idx]

    resp_x_T1 = [row for i, row in enumerate(x_coord_T1) if i in resp_idx]
    resp_y_T1 = [row for i, row in enumerate(y_coord_T1) if i in resp_idx]

    resp_x_T2 = [row for i, row in enumerate(x_coord_T2) if i in resp_idx]
    resp_y_T2 = [row for i, row in enumerate(y_coord_T2) if i in resp_idx]

else:
    #non-resp
    non_resp_x_T0 = [row for i, row in enumerate(x_coord_T0) if i in non_resp_idx]
    non_resp_y_T0 = [row for i, row in enumerate(y_coord_T0) if i in non_resp_idx]

    non_resp_x_T1 = [row for i, row in enumerate(x_coord_T1) if i in non_resp_idx]
    non_resp_y_T1 = [row for i, row in enumerate(y_coord_T1) if i in non_resp_idx]

    non_resp_x_T2 = [row for i, row in enumerate(x_coord_T2) if i in non_resp_idx]
    non_resp_y_T2 = [row for i, row in enumerate(y_coord_T2) if i in non_resp_idx]

    #resp
    resp_x_T0 = [row for i, row in enumerate(x_coord_T0) if i in resp_idx]
    resp_y_T0 = [row for i, row in enumerate(y_coord_T0) if i in resp_idx]

    resp_x_T1 = [row for i, row in enumerate(x_coord_T1) if i in resp_idx]
    resp_y_T1 = [row for i, row in enumerate(y_coord_T1) if i in resp_idx]

    resp_x_T2 = [row for i, row in enumerate(x_coord_T2) if i in resp_idx]
    resp_y_T2 = [row for i, row in enumerate(y_coord_T2) if i in resp_idx]

    #rem
    rem_x_T0 = [row for i, row in enumerate(x_coord_T0) if i in rem_idx]
    rem_y_T0 = [row for i, row in enumerate(y_coord_T0) if i in rem_idx]

    rem_x_T1 = [row for i, row in enumerate(x_coord_T1) if i in rem_idx]
    rem_y_T1 = [row for i, row in enumerate(y_coord_T1) if i in rem_idx]

    rem_x_T2 = [row for i, row in enumerate(x_coord_T2) if i in rem_idx]
    rem_y_T2 = [row for i, row in enumerate(y_coord_T2) if i in rem_idx]


def plot_points():
#plot the vector space
    plt.figure("Responder plot")
    plt.scatter(non_resp_x_T0, non_resp_y_T0, label='T0', color="#1f77b4")
    plt.scatter(non_resp_x_T1, non_resp_y_T1, label='T1', color="#2ca02c")
    # plt.scatter(non_resp_x_T2, non_resp_y_T2, label='T0', color="#d62728")

    plt.figure("Non-Responder plot")
    plt.scatter(resp_x_T0, resp_y_T0, label='T0', color="#1f77b4")
    plt.scatter(resp_x_T1, resp_y_T1, label='T1', color="#2ca02c")
    # plt.scatter(resp_x_T2, resp_y_T2, label='T0', color="#d62728")
    plt.show()

    non_resp_x_diff = np.array(non_resp_x_T1)-np.array(non_resp_x_T0)
    non_resp_y_diff = np.array(non_resp_y_T1)-np.array(non_resp_y_T0)

    resp_x_diff = np.array(resp_x_T1)-np.array(resp_x_T0)
    resp_y_diff = np.array(resp_y_T1)-np.array(resp_y_T0)

    non_resp_x_norm = np.sqrt(np.sum(non_resp_x_diff**2))
    non_resp_y_norm = np.sqrt(np.sum(non_resp_y_diff**2))

    resp_x_norm = np.sqrt(np.sum(resp_x_diff**2))
    resp_y_norm = np.sqrt(np.sum(resp_y_diff**2))

    print(f"L2-norm of non-resp\nPEn: {non_resp_x_norm}\nLMC: {non_resp_y_norm}")
    print(f"L2-norm of resp\nPEn: {resp_x_norm}\nLMC: {resp_y_norm}")

    print(f"Sum of diff non-resp\nPEn: {np.sum(non_resp_x_diff)}\nLMC: {np.sum(non_resp_y_diff)}")
    print(f"Sum of diff resp\nPEn: {np.sum(resp_x_diff)}\nLMC: {np.sum(resp_y_diff)}")

def plot_CECP_Temp():
    color =["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # plt.figure('Responder Plot')
    #resp
    x_T0 = np.mean(resp_x_T0,axis=0)
    y_T0 = np.mean(resp_y_T0, axis=0)

    x_T1 =np.mean(resp_x_T1,axis=0)
    y_T1 = np.mean(resp_y_T1, axis=0)

    x_T2 = np.mean(resp_x_T2,axis=0)
    y_T2=np.mean(resp_y_T2,axis=0)

    plt.scatter(np.mean(resp_x_T0,axis=0), np.mean(resp_y_T0, axis=0), label='T0', marker='o', facecolors='none', edgecolors=color[3])
    plt.scatter(np.mean(resp_x_T1,axis=0), np.mean(resp_y_T1, axis=0), label='T1', marker='^', color=color[2])
    plt.scatter(np.mean(resp_x_T2,axis=0), np.mean(resp_y_T2,axis=0), label='T2', marker='o', color=color[0])

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
    plt.title(f"Trend in Responder Group ({region})")


    #non-resp
    # plt.figure('Non-Responder Plot')
    x_T0_2 = np.mean(non_resp_x_T0,axis=0)
    y_T0_2 = np.mean(non_resp_y_T0, axis=0)

    x_T1_2 =np.mean(non_resp_x_T1,axis=0)
    y_T1_2 = np.mean(non_resp_y_T1, axis=0)

    x_T2_2 = np.mean(non_resp_x_T2,axis=0)
    y_T2_2 = np.mean(non_resp_y_T2,axis=0)

    plt.scatter(np.mean(non_resp_x_T0,axis=0), np.mean(non_resp_y_T0, axis=0), label='T0', marker='o', facecolors='none', edgecolors=color[3])
    plt.scatter(np.mean(non_resp_x_T1,axis=0), np.mean(non_resp_y_T1, axis=0), label='T1', marker='^', color=color[2])
    plt.scatter(np.mean(non_resp_x_T2,axis=0), np.mean(non_resp_y_T2,axis=0), label='T2', marker='o', color=color[0])

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
    plt.title(f"Trend in Non-Responder Group ({region})")
    # plt.savefig("resp_trend_temporal_UBC.png", dpi=300)

    #rem
    # plt.figure('Remission Plot')
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
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, alpha=0.6))
        
        # Arrow from T1 -> T2
    plt.annotate('', 
                    xy=(x_rem_T2, y_rem_T2),      # Arrow head target
                    xytext=(x_rem_T1, y_rem_T1),  # Arrow tail start
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5, alpha=0.6))
    plt.xlabel("PEn")
    plt.ylabel("LMC (Statistical complexity")
    plt.title(f"Trend in Ren Group ({region})")

    plt.show()
    

def plot_paired_slopegraph(t0_data, t1_data, title="Non-Responders: Metric Shift (T0 vs T1)"):
    fig, ax = plt.subplots(figsize=(5, 6))
    
    n_subjects = len(t0_data)
    
    # 1. Plot individual subject trajectories
    for i in range(n_subjects):
        y0, y1 = t0_data[i], t1_data[i]
        
        # Color line based on directional change
        if y1 > y0:
            color = '#2b5c8f'  # Blue for increase
        elif y1 < y0:
            color = '#d95f02'  # Orange/Red for decrease
        else:
            color = '#7570b3'  # Neutral
            
        ax.plot([0, 1], [y0, y1], color=color, alpha=0.35, linewidth=1.5, marker='o', markersize=5)

    # 2. Plot Group Mean Trajectory (Thick line with bold markers)
    mean_t0 = np.mean(t0_data)
    mean_t1 = np.mean(t1_data)
    
    ax.plot([0, 1], [mean_t0, mean_t1], color='black', linewidth=3.5, 
            marker='o', markersize=9, label='Group Mean', zorder=5)

    # 3. Add Mean Value Labels
    ax.text(-0.08, mean_t0, f'{mean_t0:.2f}', va='center', ha='right', fontweight='bold', fontsize=10)
    ax.text(1.08, mean_t1, f'{mean_t1:.2f}', va='center', ha='left', fontweight='bold', fontsize=10)

    # 4. Formatting & Cleanup
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['T0', 'T2'], fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    
    # Clean up spines (classic slopegraph look)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add light vertical background axes for T0 and T1
    ax.axvline(0, color='gray', linestyle='--', alpha=0.3, zorder=0)
    ax.axvline(1, color='gray', linestyle='--', alpha=0.3, zorder=0)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), frameon=False)
    plt.tight_layout()
    plt.show()


def plot_direction(title, time: tuple):
    plt.figure(figsize=(6, 6))
    plt.title(title)
    
    perm1, perm2 = time[0]
    stat1, stat2 = time[1]
    
    # Calculate vector components (deltas)
    vec_x0 = np.array(perm2) - np.array(perm1)
    vec_y0 = np.array(stat2) - np.array(stat1)
    
    # Origin coordinates (all zeros)
    origin_x = np.zeros_like(vec_x0)
    origin_y = np.zeros_like(vec_y0)

    # Calculate magnitudes
    magnitudes = np.hypot(vec_x0, vec_y0)
    thresh = 0.03
    colors = np.where(magnitudes <= thresh, '#FF6B6B', '#4D96FF')

    # Reference crosshairs at origin
    plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='grey', linewidth=0.8, linestyle='--')

    # Plot vectors starting from (0,0)
    plt.quiver(
        origin_x, origin_y, vec_x0, vec_y0, 
        angles='xy', scale_units='xy', scale=1, 
        color=colors
    )
    
    plt.xlim(-0.15, 0.15)
    plt.ylim(-0.15, 0.18)
    plt.ylabel(r"$\Delta$ LMC")
    plt.xlabel(r"$\Delta$ PEn")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def CECP_plot(title, data: tuple, data2: tuple, data3: tuple, points=True):
    x0, y0 = data[0]
    x1, y1 = data[1]
    x2, y2 = data[2]

    x_0, y_0 = data2[0]
    x_1, y_1 = data2[1]
    x_2, y_2 = data2[2]

    x_r0, y_r0 = data3[0]
    x_r1, y_r1 = data3[1]
    x_r2, y_r2 = data3[2]

    dx=7

    min_comp = ordpy.minimum_complexity_entropy(dx=6)
    max_comp = ordpy.maximum_complexity_entropy(dx=6)
    H_min = min_comp[:,0]
    H_max = max_comp[:,0]
    C_min = min_comp[:,1]
    C_max = max_comp[:,1]

    #synthetic signal
    #white noise (Gaussian)
    n_sample = 1000
    mean = 0.0
    std = 1.0
    noise_gauss = np.random.normal(mean, std, n_sample)
    H_wn, C_wn = ordpy.complexity_entropy(data=noise_gauss, dx=3)


    #plotting
    plt.figure(figsize=(7, 4.5))
    plt.plot(H_min, C_min, 'k--', label='$C_{\mathrm{min}}$')
    plt.plot(H_max, C_max, 'k-', label='$C_{\mathrm{max}}$')

    #plot the EEG points
    if points:
        plt.scatter(x0, y0, alpha=0.7, label='T0 Resp', color="#0eff36")
        plt.scatter(x1, y1, alpha=0.7, label='T1 Resp', color="#0e42ff")
        plt.scatter(x2, y2, alpha=0.7, label='T2 Resp', color="#0eff22")

        plt.scatter(x_0, y_0, alpha=0.7, label='T0 Non-Resp', color="#ff7f0e")
        plt.scatter(x_1, y_1, alpha=0.7, label='T1 Non-Resp', color="#0e42ff")
        plt.scatter(x_2, y_2, alpha=0.7, label='T2 Non-Resp', color="#0eff22")

        plt.scatter(x_r0, y_r0, alpha=0.7, label='T0 Rem', color="#ff0e42")
        plt.scatter(x_r1, y_r1, alpha=0.7, label='T1 Rem', color="#0e42ff")
        plt.scatter(x_r2, y_r2, alpha=0.7, label='T2 Rem', color="#0eff22")

    #plot the singular point
    else:
        #resp
        plt.scatter(np.mean(x0), np.mean(y0), alpha=0.7, label='T0', color="#ff7f0e")
        plt.scatter(np.mean(x1), np.mean(y1), alpha=0.7, label='T1', color="#0e42ff")
        plt.scatter(np.mean(x2), np.mean(y2), alpha=0.7, label='T2', color="#0eff22")

        #non-resp
        # plt.scatter(np.mean(x_0), np.mean(y_0), alpha=0.7, color="#ff7f0e")
        # plt.scatter(np.mean(x_1), np.mean(y_1), alpha=0.7, color="#0e42ff")
        # plt.scatter(np.mean(x_2), np.mean(y_2), alpha=0.7, color="#0eff22")

        #remission
        # plt.scatter(np.mean(x_r0), np.mean(y_r0), alpha=0.7, color="#ff7f0e")
        # plt.scatter(np.mean(x_r1), np.mean(y_r1), alpha=0.7, color="#0e42ff")
        # plt.scatter(np.mean(x_r2), np.mean(y_r2), alpha=0.7, color="#0eff22")

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
    """Zooming plots???"""

    plt.show()

def find_max_magnitude(initial: tuple, final: tuple, cond_idx):
    xi = initial[0]
    yi = initial[1]

    xf = final[0]
    yf = final[1]
    x_diff = np.array(xf) - np.array(xi)
    y_diff = np.array(yf) - np.array(yi)
    #calculate the L2 norm
    norm_sq = np.array(x_diff)**2 + np.array(y_diff)**2
    idx = np.argmax(norm_sq)
    max_val = np.max(norm_sq)
    return (cond_idx[idx], np.sqrt(max_val))

def return_id_mag(initial: tuple, final: tuple, cond_idx):
    xi = initial[0]
    yi = initial[1]

    xf = final[0]
    yf = final[1]
    x_diff = np.array(xf) - np.array(xi)
    y_diff = np.array(yf) - np.array(yi)
    #calculate the L2 norm
    norm_sq = np.array(x_diff)**2 + np.array(y_diff)**2

    ids = []
    val = []

    for idx, id in enumerate(cond_idx):
        # print(id, np.sqrt(norm_sq[idx]))
        ids.append(subject_ids[id])
        val.append(np.sqrt(norm_sq[idx]))
    return (ids, val)

#This method only works for d=3 (three points)
#Wrong fix it later
def plot_2d_ordinal_pattern(points):

    map = {
        (0,1,2): 1,
        (2,1,0): 2,
        (0,2,1): 3,
        (2,0,1): 4,
        (1,2,0): 5,
        (1,0,2): 6
    }

    color =["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    plt.figure('Responder Plot')
    #resp
    x_T0, y_T0 = points[0]

    x_T1, y_T1 = points[1]

    x_T2, y_T2 = points[2]

    plt.scatter(x_T0, y_T0, label='T0', marker='o', facecolors='none', edgecolors=color[3])
    plt.scatter(x_T1, y_T1, label='T1', marker='^', color=color[2])
    plt.scatter(x_T2, y_T2, label='T2', marker='o', color=color[0])

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
    plt.title(f"Trend in individual group ({region})")
    plt.show(block=True)


if __name__ == "__main__":
    # plot_paired_slopegraph(t0_data=resp_x_T0, t1_data=resp_x_T2)
    plot_direction(time=((rem_x_T0, rem_x_T2), (rem_y_T0, rem_y_T2)), title="vector plots of the Remission group (CAMH-Temporal)")
    CECP_plot(title='CECP of Responder Group (Temporal)', 
              data=((resp_x_T0, resp_y_T0), 
                    (resp_x_T1, resp_y_T1), 
                    (resp_x_T2, resp_y_T2)),
              data2=((non_resp_x_T0, non_resp_y_T0), 
                    (non_resp_x_T1, non_resp_y_T1), 
                    (non_resp_x_T2, non_resp_y_T2)),
              data3=((rem_x_T0, rem_y_T0), 
                    (rem_x_T1, rem_y_T1), 
                    (rem_x_T2, rem_y_T2)),
                    points=True)
    # id, max_val = find_max_magnitude(initial=(rem_x_T0, rem_y_T0), final=((rem_x_T1, rem_y_T1)))
    # print(f"ID: {id} Magnitude: {max_val}")
    """Get magnitude"""
    # ids, vals = return_id_mag(initial=(rem_x_T0, rem_y_T0), final=((rem_x_T1, rem_y_T1)), cond_idx=rem_idx)
    plot_CECP_Temp()

    """Individual plot"""
    # pts_T0 = zip(rem_x_T0, rem_y_T0)
    # pts_T1 = zip(rem_x_T1, rem_y_T1)
    # pts_T2 = zip(rem_x_T2, rem_y_T2)

    # for p0,p1,p2 in zip(pts_T0, pts_T1, pts_T2):
    #     # print(f"({p0}), ({p1}), ({p2})")
    #     print(plot_2d_ordinal_pattern(points=(p0,p1,p2)))
    

    


label = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
1, 0, 1, 1, 0, 1, 0, 0, 0, 1,
0, 0, 1, 1, 0, 0, 1, 0, 0, 0,
0, 1, 0, 1, 1, 0, 0, 0, 0, 0,
0, 1, 0, 1, 0, 0, 0, 1, 0, 0,
0, 1, 0, 0, 1, 0, 0, 0, 1, 0,
1, 0, 0, 0, 0, 0, 1, 1, 0, 0,
0]

rem = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
1, 0, 1, 1, 0, 1, 0, 0, 0, 1,
0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
0]
