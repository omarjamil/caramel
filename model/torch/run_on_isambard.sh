#!/bin/bash

# Start by copying relevant files over
rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress caramel.py train.py model.py data_io.py mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/code/torch
rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress caramel_stacked.py train_stacked.py data_io_batched.py mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/code/torch
rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress caramel_diff.py train_diff.py mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/code/torch
rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress caramel_diff_ae.py train_diff_ae.py mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/code/torch
rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress caramel_diff_multiout.py mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/code/torch
# rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress caramel_siren.py train_siren.py siren.py data_io.py mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/code/torch
# rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress caramel_pca.py train_pca.py model.py data_io.py mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/code/torch
# rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress caramel_resnet.py train_resnet.py model.py data_io.py mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/code/torch
# rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress caramel_vae.py train_vae.py model.py data_io.py mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/code/torch
# rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress caramel_ae.py train_vae.py model.py data_io.py mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/code/torch
rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress isambard_run.sub mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/code/torch

# Copy data
# rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress /project/spice/radiation/ML/CRM/data/models/datain/cnn_train_data*.hdf5 mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/data
# rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress /project/spice/radiation/ML/CRM/data/models/datain/cnn_test_data*.hdf5 mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/data
rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress /project/spice/radiation/ML/CRM/data/models/datain/train_data_023001AQT.hdf5 mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/data
rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress /project/spice/radiation/ML/CRM/data/models/datain/test_data_023001AQT.hdf5 mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/data
rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress /project/spice/radiation/ML/CRM/data/models/datain/train_data_023001AQTS.hdf5 mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/data
rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress /project/spice/radiation/ML/CRM/data/models/datain/test_data_023001AQTS.hdf5 mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/data
# rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress /project/spice/radiation/ML/CRM/data/models/datain/test_data_021501AQT.hdf5 mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/data
# rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress /project/spice/radiation/ML/CRM/data/models/datain/train_data_021501AQT.hdf5 mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/data
rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress /project/spice/radiation/ML/CRM/data/models/normaliser/ mo-ojamil@login.isambard:/home/mo-ojamil/ML/CRM/data/normaliser

# Now execute the model
ssh mo-ojamil@login.isambard "qsub /home/mo-ojamil/ML/CRM/code/torch/isambard_run.sub"

