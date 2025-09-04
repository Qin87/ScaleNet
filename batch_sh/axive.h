#!/bin/bash

net_values="LargeScaleNet"
layer_values="6"
Dir="0  0.5 1 -1"
Patiences=(400)
Normalize_vals=(0)   # 1

while pgrep -x python3 > /dev/null; do
  echo "Waiting for all python3 processes to finish..."
  sleep 180
done

# 'citeseer_npz/' 'cora_ml/'  'telegram/' 'dgl/pubmed' 'WikiCS/'
#  'WikipediaNetwork/squirrel'  'WikipediaNetwork/chameleon'  'directed-roman-empire/'
Direct_dataset=('directed-roman-empire/' )
generate_timestamp() {
  date +"%d%H%Ms%S"
}
timestamp=$(generate_timestamp)
# Iterate over each dataset
for Didataset in "${Direct_dataset[@]}"; do
      for patience in "${Patiences[@]}"; do
    for normalize in "${Normalize_vals[@]}"; do
    for BN in "${Normalize_vals[@]}"; do
    for sloop in "${Normalize_vals[@]}"; do
    for zero in "${Normalize_vals[@]}"; do
    for layer in $layer_values; do
        logfile="outforlayer${layer}.log"
        exec > "$logfile" 2>&1  # Redirect stdout and stderr to log file
           for alphadir in $Dir; do
       for betadir in $Dir; do
        for gamadir in $Dir; do
        for net in $net_values; do
            log_output="${Didataset//\//_}_${timestamp}_A${a}_alpha${dir}__${net}_layer${layer}.log"

            # Run the Python script with parameters and log output
python3 main_lit.py  --seed=111 --alphaDir="$alphadir"  --betaDir="$betadir"   --gamaDir="$gamadir"  --num_split=1   \
--NotImproved="$patience"  --use_best_hyperparams=1     --weight_penalty='None'\
  --normalize="$normalize"   --zero_order="$zero"  --BN_model="$BN"  --add_selfloop="$sloop"  --dropout=0.0  --all1=1 \
 --net="$net"  --layer="$layer"   --Dataset="$Didataset" >  \
"$log_output"
             2>&1
            wait $pid
        done
       done
       done
       done
       done
       done
done
       done
       done
done
done