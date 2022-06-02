import os 
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import torch
from sklearn.preprocessing import normalize
import torchvision.transforms as T
from PIL import Image
from utils import pc_normalize, random_select_points, shift_point_cloud, \
    jitter_point_cloud, generate_random_rotation_matrix, \
    generate_random_tranlation_vector, transform

class Objects(Dataset):
    def __init__(self,dataset_dir, transform=None, target_transform=None):
        self.dataset_dir  = dataset_dir
        self.objects = os.listdir(self.dataset_dir)



    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        object = self.objects[idx]
        obj_path = os.path.join(self.dataset_dir,object)    
        files = os.listdir(obj_path)
        files = [file  for file in files if (file.endswith(".npy") and file != 'all.npy') ] 

        # print(files)
        num = len(files)
        frag_id = np.random.randint(num)
        frag = files[frag_id]
        frag_data = np.load(os.path.join(obj_path,frag))[:,:3]
        R, t = generate_random_rotation_matrix(-1, 1), generate_random_tranlation_vector(-0.5, 0.5)
        frag_data = transform(frag_data, R, t)
        arr = np.zeros((1,6))
        for file in files:
  
            if file !=frag:
                data = np.load(os.path.join(obj_path,file))
                arr = np.vstack((arr,data))
        arr = np.delete(arr,0,0)
        data = {}
        data['frag'] = torch.from_numpy(frag_data)
        data['other'] = torch.from_numpy(arr[:,:3])
        data['all'] = torch.from_numpy(np.load(os.path.join(obj_path,'all.npy'))[:,:3])
        return data,R,t

