#!/bin/bash

net_values=" ScaleNet  "
q_value=0
layer_values=" 4 "
imbal="2  "
Dir="0 0.5 1 "

# 'citeseer_npz/' 'cora_ml/'  'telegram/' 'dgl/pubmed' 'WikiCS/'
# 'WikipediaNetwork/squirrel'  'WikipediaNetwork/chameleon'
Direct_dataset=( 'telegram/' )
Direct_dataset_filename=$(echo $Direct_dataset | sed 's/\//_/g')
generate_timestamp() {
  date +"%d%H%Ms%S"
}
timestamp=$(generate_timestamp)

# Iterate over each dataset # --alphaDir="$alphadir"  --betaDir="$betadir" --gamaDir="$gamadir"
for Didataset in "${Direct_dataset[@]}"; do
    for layer in $layer_values; do
        logfile="outforlayer${layer}.log"
        exec > "$logfile" 2>&1  # Redirect stdout and stderr to log file
       for alphadir in $Dir; do
        # for betadir in $Dir; do
        #for gamadir in $Dir; do
        for net in $net_values; do
            log_output="${Didataset//\//_}_${timestamp}_A${a}_bataDir${alphadir}__${net}_layer${layer}q${q_value}.log"

            # Run the Python script with parameters and log output
python3 main.py   --differ_AA=1    --use_best_hyperparams=0   --num_split=1  --alphaDir="$alphadir"    --net="$net"  --layer="$layer" \
--Dataset="$Didataset" > "$log_output"  \
             2>&1
            wait $pid
          #done
       # done
      # done
       done
       done
       done
done