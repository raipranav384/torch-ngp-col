import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
import json
import os
import cv2
# Define a Dataset class that inherits from torch.utils.data.Dataset

class CamDataset(Dataset):
    def __init__(self,json_path ,transform=None):
        self.transform = transform
        
        self.parent_dir=os.path.dirname(json_path)

        with open(json_path) as f:
            self.json_data = json.load(f)
        fx=self.json_data['fl_x']
        fy=self.json_data['fl_y']
        cx=self.json_data['cx']
        cy=self.json_data['cy']
        self.frames=self.json_data['frames']
        if 'aabb_scale' in self.json_data:
            self.aabb=self.json_data['aabb_scale']
        self.K=np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ]
        ).astype(np.float32)
        self.frames=self.frames
        self.K_homo=torch.eye(4)
        self.K=torch.tensor(self.K)
        self.K_homo[:3,:3]=self.K
        self.K_inv=torch.linalg.inv(self.K_homo)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        sample = self.frames[idx]
        img_path=os.path.join(self.parent_dir,sample['file_path'])
        img=cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        C2W=torch.tensor(np.array(sample['transform_matrix'])).to(torch.float32)
        W2C=torch.linalg.inv(C2W)
        W2I=(self.K_homo@W2C)        

        img=torch.from_numpy(img).float()
        # img_mask=img.sum(axis=-1,keepdim=True)
        img=img.permute(2,0,1)/255.0
        
        # if self.transform:
        #     sample = self.transform(sample)
        return img,W2I,W2C,self.K_homo
    
if __name__=='__main__':
    camData=CamDataset(json_path='/home/chronos/solid/face_data/transforms_train.json')
    for i in range(len(camData)):
        img,_,W2C,_=camData[i]
        print(torch.linalg.det(W2C[:3,:3]))
