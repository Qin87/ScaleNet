#!/bin/bash

net_values=" gen_GCN   "    #
layer_values="  5 "

# 'directed-roman-empire/' 'telegram/'  'citeseer/' 'cora_ml/'   'WikiCS/'
# 'citeseer/' 'cora_ml/'  'telegram/'   'dgl/pubmed'  'WikiCS/'  'WikipediaNetwork/chameleon' 'WikipediaNetwork/squirrel'
Direct_dataset=(  'citeseer/' 'cora_ml/'  'telegram/'   'dgl/pubmed'  'WikiCS/'  'WikipediaNetwork/chameleon' 'WikipediaNetwork/squirrel' )
generate_timestamp() {
  date +"%d%H%Ms%S"
}
timestamp=$(generate_timestamp)

while pgrep -f main.py >/dev/null || pgrep -f main.py >/dev/null; do
    sleep 30
done

# Iterate over each dataset   --net="$net"    --layer="$layer"
for Didataset in "${Direct_dataset[@]}"; do
     for layer in $layer_values; do
        logfile="outforlayer${layer}.log"
        exec > "$logfile" 2>&1  # Redirect stdout and stderr to log file
        for net in $net_values; do
            log_output="${Didataset//\//_}_${timestamp}_${net}_dropout.log"

            # Run the Python script with parameters and log output
            python3 main.py    --net="$net"   --layer="$layer"  --add_selfloop=0 --dropout=0.2 --to_undirected=1  \
            --Dataset="$Didataset" > "$log_output"
             2>&1
            wait $pid
           done
        done
done