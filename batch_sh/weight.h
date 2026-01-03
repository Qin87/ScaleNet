#!/bin/bash

net_values="GAT "
layer_values="1 2 3 4 5"
Dir="0  0.5 1 -1"
Patiences=(400)
Normalize_vals=(0)   # 1


# 'citeseer/' 'cora_ml/'  'telegram/' 'dgl/pubmed' 'WikiCS/'
#  'WikipediaNetwork/squirrel'  'WikipediaNetwork/chameleon'  'directed-roman-empire/'   'arxiv-year/'
Direct_dataset=('Cora' 'CiteSeer'   )
generate_timestamp() {
  date +"%d%H%Ms%S"
}
timestamp=$(generate_timestamp)


for Didataset in "${Direct_dataset[@]}"; do
    for layer in $layer_values; do
        logfile="outforlayer${layer}.log"
        exec > "$logfile" 2>&1  # Redirect stdout and stderr to log file
        for net in $net_values; do
            log_output="${Didataset//\//_}_${timestamp}_A${a}_alpha${dir}__${net}_layer${layer}.log"

            # Run the Python script with parameters and log output
python3 main_lit.py  --seed=0   --num_split=10   \
 --net="$net"  --layer="$layer"   --Dataset="$Didataset" >  \
"$log_output"
             2>&1
            wait $pid
        done
       done
       done
