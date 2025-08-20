#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D /users/username
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
#SBATCH -o job-%j.output
#SBATCH -e job-%j.error
## Job name
#SBATCH -J gpu-test
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=6-23:05:00
## Memory limit (in megabytes). Total --mem or amount per cpu --mem-per-cpu
#SBATCH --mem-per-cpu=802400
## GPU requirements
#SBATCH --gres gpu:1
## Specify partition
#SBATCH -p gpu

################# Part-2 Shell script ####################
#===============================
#  Activate Flight Environment
#-------------------------------
source "${flight_ROOT:-/opt/flight}"/etc/setup.sh

#==============================
#  Activate Package Ecosystem
#------------------------------
flight env activate conda@Apr15


#===========================
#  Create results directory
#---------------------------
RESULTS_DIR="$(pwd)/${SLURM_JOB_NAME}-outputs/${SLURM_JOB_ID}"
echo "Your results will be stored in: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

#===============================
#  Application launch commands
#-------------------------------
# Customize this section to suit your needs.
net_values="QiGi2 QiGi3 QiGu3 QiGu2 QiGu4 QiGi4"
q_value=0
layer_values="1 2 3  4  "    #

# 'citeseer_npz/' 'cora_ml/'  'telegram/telegram'
Direct_dataset=( 'citeseer_npz/' 'cora_ml/'  'telegram/telegram' 'dgl/pubmed' 'dgl/cora' 'WikiCS/'  )  # 'cora_ml/'  'citeseer_npz/'  'WebKB/Cornell' 'WebKB/wisconsin'  'WebKB/texas' 'WebKB/texas' 'WebKB/wisconsin'  telegram/telegram
Direct_dataset_filename=$(echo $Direct_dataset | sed 's/\//_/g')
generate_timestamp() {
  date +"%d%H%Ms%S"
}
timestamp=$(generate_timestamp)

# Iterate over each dataset
for Didataset in "${Direct_dataset[@]}"; do
    for layer in $layer_values; do
        logfile="outforlayer${layer}.log"
        exec > "$logfile" 2>&1  # Redirect stdout and stderr to log file
        for net in $net_values; do
            # Run the Python script with parameters and log output
            python3 main.py --net="$net"  --W_degree=5  --layer="$layer" --q="$q_value" --Dataset="$Didataset" > "NormQym_NoSelfLoop_${Didataset//\//_}_${timestamp}_${net}_layer${layer}q${q_value}.log"

            wait $pid
          done
        done
done
