#!/bin/bash

net_values=" LargeScaleNet  ScaleNet"
layer_values="  2 "


# 'citeseer/' 'cora_ml/'  'telegram/'   'dgl/pubmed'  'WikiCS/'  'WikipediaNetwork/chameleon' 'WikipediaNetwork/squirrel'
Direct_dataset=( 'WikiCS/'  )
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
            log_output="${Didataset//\//_}_${timestamp}_${net}_layer${layer}.log"

            # Run the Python script with parameters and log output
            python3 main.py    --net="$net"   --layer="$layer" \
            --Dataset="$Didataset" > "$log_output"
             2>&1
            wait $pid
          # done
        done
        done
done