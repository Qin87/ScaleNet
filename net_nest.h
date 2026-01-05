#!/bin/bash

net_values="RiG  "
layer_values=" 2 3 4  "
incinorm="dir sym row 0 "
lr0="0.1 0.05 0.01 0.005"

# 'citeseer/' 'cora_ml/'  'telegram/'   'dgl/pubmed'  'WikiCS/'   'WikiCS/' 'WikipediaNetwork/squirrel'
# 'WikipediaNetwork/chameleon'   'telegram/'   'dgl/pubmed'  'citeseer/'   #"$inci"
Direct_dataset=(   'Cora'   )
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
          for inci  in $incinorm; do
          for lr  in $lr0; do
        for net in $net_values; do
            log_output="${Didataset//\//_}_${timestamp}_${net}_layer${layer}.log"

            python main.py   --net="$net"  --layer="$layer"  --use_best_hyperparams=0 --num_split=10   --inci_norm="$inci"  --lr="$lr" \
            --Dataset="$Didataset" > "$log_output"
             2>&1
            wait $pid
           done
        done
        done
done
        done
        done
done