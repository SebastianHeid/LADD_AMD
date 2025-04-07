CONFIG_PATH='/export/home/sheid/LADD_AMD/config/config_eval_8.yaml'
available_gpus=(5)
for gpu in "${available_gpus[@]}"; do
CUDA_VISIBLE_DEVICES=${gpu} python3 core/evaluation/create_generated_dataset.py --config_path ${CONFIG_PATH} 
done

wait
for gpu in "${available_gpus[@]}"; do
CUDA_VISIBLE_DEVICES=${gpu} python3 core/evaluation/evaluation_FID.py --config_path ${CONFIG_PATH} 
done

wait
for gpu in "${available_gpus[@]}"; do
CUDA_VISIBLE_DEVICES=${gpu} python3 core/evaluation/evaluation_CLIP.py --config_path ${CONFIG_PATH} 
done