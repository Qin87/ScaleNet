#!/bin/bash

net_values="GCN"
layer_values="3"
incinorm=" 0 "   # row   sym   0  dir

while pgrep -f main_lit.py >/dev/null || pgrep -f main.py >/dev/null; do
    sleep 30
done

# 'citeseer/' 'cora_ml/'  'telegram/' 'dgl/pubmed' 'WikiCS/'
#  'WikipediaNetwork/squirrel'  'WikipediaNetwork/chameleon'
Direct_dataset=( 'WikiCS/'  )
generate_timestamp() {
  date +"%d%H%Ms%S"
}
timestamp=$(generate_timestamp)
# Iterate over each dataset
for Didataset in "${Direct_dataset[@]}"; do
# for att0 in "${att[@]}"; do
    for layer in $layer_values; do
      for inci  in $incinorm; do

        logfile="outforlayer${layer}.log"
        exec > "$logfile" 2>&1  # Redirect stdout and stderr to log file
        for net in $net_values; do
            log_output="${Didataset//\//_}_${timestamp}_A${a}_alpha${dir}__${net}_layer${layer}.log"

            # Run the Python script with parameters and log output
python3 main.py  --seed=12  --num_split=10     --inci_norm="$inci" --add_selfloop=1  --hid_dim=512\
 --use_best_hyperparams=0    \
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
done