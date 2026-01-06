#!/bin/bash

# Wait until no python / python3 process is running for this user
while pgrep -u "$USER" -f "python" > /dev/null; do
    sleep 2000
done

# Shutdown the system
sudo shutdown -h now
