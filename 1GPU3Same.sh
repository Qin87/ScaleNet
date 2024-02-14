#!/bin/bash

# List of AugDirect values
augdirect_values="0 1 -1 2 4 20 21 22 23 231 2311"
Direct_dataset='cora_ml/'  # Set your Direct_dataset value
Direct_dataset_filename=$(echo $Direct_dataset | sed 's/\//_/g')
IsDirData=False
unDirect_data='CiteSeer'
net='DiG'

generate_timestamp() {
  date +"%d%H%M%S"
}
timestamp=$(generate_timestamp)

logfile=$random_digit_"outfor2.log"
exec > $logfile 2>&1

# Function to wait for PIDs
wait_for_pids() {
  for pid in "$@"; do
    wait $pid
  done
}

# Iterate over each AugDirect value
for augdirect in $augdirect_values; do
  # Run two processes concurrently on GPU 0
  for ((i=0; i<3; i++)); do
    GPUdevice=0  # Set the GPU device to 0
    if [ "$IsDirData" = False ]; then
      filename="${net}_${unDirect_data}"
    else
      filename="${net}_${Direct_dataset_filename}"
    fi

    # Run the process in the background
    nohup python DiGMain.py --GPUdevice=$GPUdevice --AugDirect=$augdirect  --net=$net \
      --Direct_dataset=$Direct_dataset --undirect_dataset=$unDirect_data \
      > ${filename}_Aug${augdirect}_T${timestamp}_GPU${GPUdevice}_${i}.log &
    
    # Store the last PID for GPU 0
    last_pid["$GPUdevice"]=$!
  done
  
  # Wait for the two processes on GPU 0 to finish before moving to the next iteration
  wait_for_pids "${last_pid[@]}"
done
