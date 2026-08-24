#!/bin/bash
#SBATCH --account=def-sblain
#SBATCH --cpus-per-task=20          
#SBATCH --mem=24G                   
#SBATCH --time=12:00:00             
#SBATCH --job-name=neuro_analysis
#SBATCH --output=logs/%x-%j.out      
#SBATCH --error=logs/%x-%j.err      

#setting up the env for running the code
module load python/3.11 
module load scipy-stack
source /home/slee172/projects/def-sblain/slee/BIAPT_Lab/bin/activate


#run the script
echo "Starting analysis at $(date)"

python /home/slee172/projects/def-sblain/slee/BIAPT_Lab/PICU_Criticality_Prognosis/scripts/feature_extraction/preprocessing_CAMH_v2.py

echo "Analysis finished at $(date)"