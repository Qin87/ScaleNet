#!/bin/bash

net_values="GCN  Dir-GNN   LargeScaleNet FaberNet"   #  Dir-GNN   LargeScaleNet FaberNet
layer_values="  2 "

# 'arxiv-year/') #  'directed-roman-empire/')  #
# 'citeseer/' 'cora_ml/'  'telegram/'   'dgl/pubmed'  'WikiCS/'  'WikipediaNetwork/chameleon' 'WikipediaNetwork/squirrel'
Direct_dataset=( 'telegram/') #  'citeseer/' 'cora_ml/'   'WikiCS/'  'WikipediaNetwork/chameleon' 'WikipediaNetwork/squirrel'  )
generate_timestamp() {
  date +"%d%H%Ms%S"
}
timestamp=$(generate_timestamp)

while pgrep -x python3 > /dev/null; do
  echo "Waiting for all python3 processes to finish..."
  sleep 10
done

# Iterate over each dataset   --net="$net"    --layer="$layer"
for Didataset in "${Direct_dataset[@]}"; do
     for layer in $layer_values; do
        logfile="outforlayer${layer}.log"
        exec > "$logfile" 2>&1  # Redirect stdout and stderr to log file
        for net in $net_values; do
            log_output="${Didataset//\//_}_${timestamp}_${net}_dropout.log"

            # Run the Python script with parameters and log output
            python3 main.py    --net="$net"   --layer="$layer"  --add_selfloop=1   \
            --Dataset="$Didataset" > "$log_output"
             2>&1
            wait $pid
           done
        done
done