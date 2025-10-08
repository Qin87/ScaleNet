#!/bin/bash

net_values="LargeScaleNet "
layer_values=" 8"
aDir="0  0.5 1 -1"
bDir="0  0.5 1 -1"
cDir="0  0.5 1 -1"
Patiences=(400)

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
for Didataset in "${Direct_dataset[@]}"; do
      # for patience in "${Patiences[@]}"; do
    for layer in $layer_values; do
        logfile="outforlayer${layer}.log"
        exec > "$logfile" 2>&1  # Redirect stdout and stderr to log file
         for alphadir in $aDir; do
       for betadir in $bDir; do
        for gamadir in $cDir; do
        for net in $net_values; do
            log_output="${Didataset//\//_}_${timestamp}_A${a}_alpha${dir}__${net}_layer${layer}.log"

python3 main_lit.py   --alphaDir="$alphadir"  --betaDir="$betadir"    --gamaDir="$gamadir" \
--num_split=1  --add_selfloop=0  \
--dropout="$gamadir"      \
--use_best_hyperparams=1   \
--weight_penalty='None'\
 --net="$net"  \
 --all1=1 \
 #--patience="$patience"  \
--Dataset="$Didataset" > "$log_output" 2>&1

            wait $pid
        done
       done
       done
       done
       done
       done
done