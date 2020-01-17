#!/bin/bash

# Start by copying relevant files over
rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress *.py mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/code/torch
rsync avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress isambard_run.sub mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/code/torch

# Copy data
rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress /project/spice/radiation/ML/CRM/data/models/datain/train_test_data_all_50S69W_std.hdf5 mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/data
rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress /project/spice/radiation/ML/CRM/data/models/std*.hdf5 mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/data/normaliser

# Now execute the model
ssh mo-ojamil@login.isambard "qsub /home/mo-ojamil/ML/CRM/code/torch/isambard_run.sub"

