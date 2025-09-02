#!/bin/bash

net_values="LargeScaleNet"
layer_values="7 8"
bDir="256"
Dir="0.01 0.05 0.1 "
Patiences=(400 )

while pgrep -x python3 > /dev/null; do
  echo "Waiting for all python3 processes to finish..."
  sleep 600
done

# 'citeseer_npz/' 'cora_ml/'  'telegram/' 'dgl/pubmed' 'WikiCS/'
#  'WikipediaNetwork/squirrel'  'WikipediaNetwork/chameleon'  'directed-roman-empire/'
Direct_dataset=('arxiv-year/' )
generate_timestamp() {
  date +"%d%H%Ms%S"
}
timestamp=$(generate_timestamp)
# Iterate over each dataset
for Didataset in "${Direct_dataset[@]}"; do
      for patience in "${Patiences[@]}"; do
    for layer in $layer_values; do
        logfile="outforlayer${layer}.log"
        exec > "$logfile" 2>&1  # Redirect stdout and stderr to log file
   #        for alphadir in $Dir; do
       for betadir in $bDir; do
        for gamadir in $Dir; do
        for net in $net_values; do
            log_output="${Didataset//\//_}_${timestamp}_A${a}_alpha${dir}__${net}_layer${layer}.log"

            # Run the Python script with parameters and log output
python3 main_lit.py   --alphaDir=0.5  --betaDir=0    --gamaDir=0  --num_split=1  --add_selfloop=0  \
--NotImproved="$patience"  --hid_dim="$betadir" --lr="$gamadir" --use_best_hyperparams=1  --all1=0  --weight_penalty='None'\
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