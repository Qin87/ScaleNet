#!/bin/bash

net_values=" GAT    "
layer_values=" 2   "
incinorm="  row "   #
lr0=" 0.005 "

while pgrep -x python3 >/dev/null; do
    sleep 30
done

# # PubMed, Coauthor-physics, , Amazon-Computers, , 'Amazon-Computers' 'Coauthor-CS'   'PubMed' 'Coauthor-CS'
# 'citeseer/' 'cora_ml/'  'telegram/'   'dgl/pubmed'  'WikiCS/'   'WikiCS/' 'WikipediaNetwork/squirrel'
# 'WikipediaNetwork/chameleon'   'telegram/'   'dgl/pubmed'  'citeseer/'   #"$inci"
Direct_dataset=(   'PubMed'    ) # 'Coauthor-CS'   'PubMed' 'Coauthor-CS'    )
Direct_dataset_filename=$(echo $Direct_dataset | sed 's/\//_/g')
generate_timestamp() {
  date +"%d%H%Ms%S"
}
timestamp=$(generate_timestamp)

# Iterate over each dataset
for Didataset in "${Direct_dataset[@]}"; do
    for layer in $layer_values; do
        logfile="outforlayer${layer}.log"
        exec > "$logfile" 2>&1  # Redirect stdout and stderr to log file
          for inci  in $incinorm; do
          for lr  in $lr0; do
        for net in $net_values; do
            log_output="${Didataset//\//_}_${timestamp}_${net}_layer${layer}.log"

            python3 main.py   --net="$net"  --layer="$layer" --nonlinear=0 \
           --use_best_hyperparams=1 --num_split=10   --inci_norm="$inci"  \
        --heads=2  --hid_dim=128 --seed=0 --originGAT=0 --r20_per_class=0  --to_undirected=1 --lr="$lr" --BN_model=0  \
            --Dataset="$Didataset" > "$log_output"
             2>&1
            wait $pid
           done
        done
        done
done
        done
        done
done