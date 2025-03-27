# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# options: 'stabilityai/stable-diffusion-2-1-base', 'PixArt-alpha/PixArt-Sigma-XL-2-1024-MS'
export MODEL_NAME='stabilityai/stable-diffusion-2-1-base'
export PROJ_NAME='add_v21_base'
export EXP_NAME='training4'
export DATA_ROOT='/export/data/vislearn/rother_subgroup/sheid/LAION_LADD/'
export WANDB_DIR='/export/data/sheid/LADD_results/wandb/training_4'


accelerate launch --mixed_precision bf16 --num_machines 1 --num_processes 1 --gpu_ids 0 train.py \
    --config_path='/export/home/sheid/LADD_AMD/config/config_training_4.yaml' \
    --base_model=$MODEL_NAME \
    --mixed_precision=bf16 \
    --G_lr=1e-6 \
    --D_lr=1e-6 \
    --max_train_steps=50000 \
    --dataloader_num_workers=0 \
    --dataset_root=$DATA_ROOT \
    --data_pkl_name='summary.pkl' \
    --validation_steps=500 \
    --checkpointing_steps=250 \
    --train_batch_size=128 \
    --gradient_checkpointing \
    --gradient_accumulation_steps=1 \
    --seed=1 \
    --project_name=${PROJ_NAME} \
    --exp_name=${EXP_NAME} \
    --zero_snr \
    --num_ts 4 \
    --ckpt_folder='/export/data/sheid/LADD_results/checkpoints/training_4' \
    --multiscale_D \
    --misaligned_pairs_D \
    --report_to=wandb \
    --project_dir='/export/home/sheid/LADD_AMD/logging' \
    --resume_from_checkpoint='/export/data/sheid/LADD_results/checkpoints/training_4/add_v21_base_training4/checkpoint-4250' \


