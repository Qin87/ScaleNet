#!/bin/bash

net_values="  ScaleNet "
layer_values=" 4   "

feat_types=("permute" "degree")   # "original"  "all1" "random"
inci_norms=("0"  "dir" "sym" "row")    #

while pgrep -x python3 > /dev/null; do
  echo "Waiting for all python3 processes to finish..."
  sleep 100
done

# 'citeseer/' 'cora_ml/'  'telegram/'   'dgl/pubmed'  'WikiCS/'     --net="$net"  'WikipediaNetwork/chameleon'
Direct_dataset=(  'directed-roman-empire/'     )
Direct_dataset_filename=$(echo $Direct_dataset | sed 's/\//_/g')
generate_timestamp() {
  date +"%d%H%Ms%S"
}
timestamp=$(generate_timestamp)



# Iterate over each dataset   --net="$net"    --layer="$layer"
for Didataset in "${Direct_dataset[@]}"; do
    for layer in $layer_values; do
        logfile="outforlayer${layer}.log"
        exec > "$logfile" 2>&1  # Redirect stdout and stderr to log file
        for net in $net_values; do
        for feat in "${feat_types[@]}"; do
        for norm in "${inci_norms[@]}"; do
            log_output="${Didataset//\//_}_${timestamp}_${net}_layer${layer}q${q_value}.log"
echo "Running: dataset=$Didataset net=$net layer=$layer feat=$feat norm=$norm"
            # Run the Python script with parameters and log output
            python3 main.py   --net="$net"     \
            --feat_type="$feat" \
            --hid_dim=1 \
            --zero_order=1\
            --inci_norm="$norm"   --num_split=1 \
             --use_best_hyperparams=1   \
            --Dataset="$Didataset" > "$log_output"
             2>&1
            wait $pid
         done
        done
        done
done
done