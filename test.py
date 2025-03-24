import os 
import pickle
from core.data.dataset import ADDDataset
print("start")
dataset_root = "/export/data/vislearn/rother_subgroup/sheid/LAION_LADD/"
data_pkl_name = "summary.pkl"
train_dataset = ADDDataset(dataset_root,
                            data_pkl_name)
print("end")