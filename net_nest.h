#!/bin/bash

# List of AugDirect values --MakeImbalance
#net_values="DiGSymib DiGSymCatib  DiG DiGib  DiGSymCatMixib"
# DiGi4 DiGu3 DiGu4
#addSym Sym addSympara
#Mag MagQin Sig Qua

#GCN GAT APPNP GIN Cheb SAGE  220 230 -2  0 1 -1 2 21 20  22 23 4
#JKNet pgnn mlp sgc"Cheb MagQin  DiGSymib DiGSymCatib  DiG DiGib  DiGSymCatMixib DiGSymCatMixSymib GAT GCN  SAGE  # QiGi4
net_values="   QiGQymu4 "
q_value=0
Aug_value=" 0 "
layer_values="2 3   "    #


Direct_dataset=( 'cora_ml/' )  # 'cora_ml/'  'citeseer_npz/'  'WebKB/Cornell' 'WebKB/wisconsin'  'WebKB/texas' 'WebKB/texas' 'WebKB/wisconsin'  telegram/telegram
Direct_dataset_filename=$(echo $Direct_dataset | sed 's/\//_/g')
unDirect_data='Cora'
generate_timestamp() {
  date +"%d%H%Ms%S"
}
timestamp=$(generate_timestamp)

# Iterate over each net value   --MakeImbalance  --IsDirectedData --to_undirected  for Didataset in $Direct_dataset; do
for Didataset in "${Direct_dataset[@]}"; do
    for Aug in $Aug_value; do
        for layer in $layer_values; do
          logfile="outforlayer${layer}.log"  # Adjust log file name with layer number
            exec > $logfile 2>&1  # Redirect stdout and stderr to log file
          # Iterate over each layer value
          for net in $net_values; do
            nohup python3 All2MainStop.py --AugDirect=$Aug --net=$net   --W_degree=1  --IsDirectedData  \
            --layer=$layer  --q=$q_value  --Direct_dataset="$Didataset"  --undirect_dataset="$unDirect_data" \
              > ${timestamp}.log &
            pid=$!

            wait $pid
          done
        done
    done
done