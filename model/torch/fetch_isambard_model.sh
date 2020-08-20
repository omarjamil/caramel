#!/bin/bash

echo "Get the latest trained models"
# file=`ssh mo-ojamil@login.isambard "ls -t /home/mo-ojamil/ML/CRM/data/models/torch | head -n 1"`
rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/data/models/torch/*.tar /project/spice/radiation/ML/CRM/data/models/torch/
rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress mo-ojamil@login.isambard:/home/mo-ojamil/caramel.* /home/h06/ojamil/analysis_tools/ML/CRM/caramel/model/torch/isambard_logs/
