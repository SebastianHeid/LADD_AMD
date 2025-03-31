CONFIG_PATH='/export/home/sheid/LADD_AMD/config/config_eval_6_4.yaml'
available_gpus=(4)
for gpu in "${available_gpus[@]}"; do
CUDA_VISIBLE_DEVICES=${gpu} python3 create_generated_dataset.py --config_path ${CONFIG_PATH} 
done

wait
for gpu in "${available_gpus[@]}"; do
CUDA_VISIBLE_DEVICES=${gpu} python3 evaluation_FID.py --config_path ${CONFIG_PATH} 
done

wait
for gpu in "${available_gpus[@]}"; do
CUDA_VISIBLE_DEVICES=${gpu} python3 evaluation_CLIP.py --config_path ${CONFIG_PATH} 
done