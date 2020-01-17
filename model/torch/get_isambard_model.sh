#!/bin/bash

echo "Get Trained model"
file=`ssh mo-ojamil@login.isambard "ls -t /home/mo-ojamil/ML/CRM/data/models/torch | head -n 1"`
rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/data/models/torch/$file /project/spice/radiation/ML/CRM/data/models/torch/
