export data_folder=${DOMINO_DATASETS_DIR}/${DOMINO_PROJECT_NAME}
export num_epochs=5
export num_trials=5
export cpus_per_trial=1
export gpus_per_trial=0

python train.py --storage_path $data_sets_folder  --num_epochs $num_epochs --num_trials $num_trials --cpus_per_trial $cpus_per_trial --gpus_per_trial $gpus_per_trial