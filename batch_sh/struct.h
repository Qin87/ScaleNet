#!/bin/bash

net_values="link linkx link_concat linkx_git "
layer_values="2 "
imbal="100  "
Binary="0 1 "

# 'citeseer/' 'cora_ml/'  'telegram/'   'dgl/pubmed'  'WikiCS/'  'WikipediaNetwork/chameleon' 'WikipediaNetwork/squirrel'
Direct_dataset=(  'fb100/penn94' )
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
         for binall1 in $Binary; do
         for binUndir in $Binary; do
         # for binReverse in $Binary; do
        for net in $net_values; do
            log_output="${Didataset//\//_}_${timestamp}_${net}_layer${layer}.log"

            # Run the Python script with parameters and log output
            python3 main.py    --net="$net"   --layer="$layer"   --all1="$binall1"   --to_undirected="$binUndir" \
            --Dataset="$Didataset" > "$log_output"
             2>&1
            wait $pid
          done
          done
        done
        done
done