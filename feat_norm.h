#!/bin/bash

net_values="  LargeScaleNet "
layer_values=" 1 2 3 4"  #  5 6  "

feat_types=("original")  #  "random" )   #    "all1"  "original" "permute" )    #   )  #    )   #
inci_norms=("dir")    #  "0"  "dir" "sym"

while pgrep -x python3 > /dev/null; do
  echo "Waiting for all python3 processes to finish..."
  sleep 100
done

# 'citeseer/' 'cora_ml/'  'telegram/'   'dgl/pubmed'  'WikiCS/'     --net="$net"  'WikipediaNetwork/chameleon'
Direct_dataset=('directed-roman-empire/'  ) #   'arxiv-year/'     )
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
            python3 main.py   --net="$net"   --layer="$layer"   --alphaDir=0  \
            --feat_type="$feat"   \
            --inci_norm="$norm"   --num_split=1 \
             --use_best_hyperparams=1  --betaDir=0.5  --gamaDir=0.5  \
            --Dataset="$Didataset" > "$log_output"
             2>&1
            wait $pid
         done
        done
        done
done
done