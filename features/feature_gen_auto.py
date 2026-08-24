import subprocess

"""
Code for running Derek's older version of PICU code for feature generations
Run this code in the ComputeCan
"""

# Your list of patient IDs
#For t0 I removed ID 732
# ids = [
#     732, 749, 752, 766, 784, 790, 792, 811, 812, 814, 
#     821, 823, 824, 828, 830, 838, 839, 847, 848, 855, 
#     856, 870, 874, 877, 879, 881, 896, 905, 906, 913, 
#     919, 920, 924
# ]

#id 3 omitted
# ids = [ '0001', '0004', '0005', '0007', '0008', '0009', '0010', 
#         '0011', '0012', '0013', '0015', '0017', '0018', '0019', '0020', 
#         '0021', '0022', '0023', '0025', '0026', '0027', '0028', '0029', 
#         '0030', '0033', '0034', '0035', '0036', '0037', '0038', '0040', 
#         '0041', '0042', '0043', '0044', '0045', '0046', '0047', '0050', 
#         '0051', '0052']

ids = ["0003"]

base_path = "/Volumes/NINET/CARTBIND_EEG_data/CARTBIND_UBC/T0_epoch"
out_dir = "/Volumes/NINET/CARTBIND_EEG_data/CARTBIND_UBC/Features/T0"
script_path = "/Users/andrewlee/BIAPT_Lab/PICU_Criticality_Prognosis/scripts/feature_extraction/neural_complexity.py"
for patient_id in ids:
    #file name + condition
    file_path = f"{base_path}/CBN02_{patient_id}_REST_EC_T0_UBC_epo.fif"
    condition = f"{patient_id}_REST_EC_T0"
    
    #running the command in the terminal
    command = [
        "python3", script_path , file_path,
        "--save",
        "--out_dir", out_dir,
        "--condition", condition,
        "--inspection",
        "--entropy",
        "--complexity",
        "--PDF"
    ]
    print(f"--- Processing ID: {patient_id} ---")
    
    #run the command
    subprocess.run(command)

    