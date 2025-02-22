#!/bin/bash

net_values="  GCNAk "
layer_values=" 1 2 3 4 5 6 7 8 9 10 11 12 13 15 16 17 18 19 20 "
#layer_values="   1  15 16 17 18 19 20 30 40 50 60 70 "   1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18

# 'citeseer/' 'cora_ml/'  'telegram/'   'dgl/pubmed'  'WikiCS/'     --net="$net"  'WikipediaNetwork/chameleon'
Direct_dataset=(   'cora_ml/'     )
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
        # for imba_value  in $imbal; do
        for net in $net_values; do
            log_output="${Didataset//\//_}_${timestamp}_${net}_layer${layer}q${q_value}.log"

            # Run the Python script with parameters and log output
            python3 main.py --num_edge=0  --net="$net"  --Ak="$layer"  --to_reverse_edge=0  --to_undirected=1  --use_best_hyperparams=1 --gcn_norm=1 \
            --Dataset="$Didataset" > "$log_output"
             2>&1
            wait $pid
          # done
        done
        done
done
