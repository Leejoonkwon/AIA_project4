import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import Dataset
import pandas as pd

class MnistDataset(Dataset):
    def __init__(self,csv_file):
        self.data_df = pd.read_csv(csv_file,header=None)
        pass
    def __len__(self):
        return len(self.data_df)
    def __getitem__(self, index):
        # 이미지 목표(label)
        label = self.data_df.iloc[index,0]
        target = torch.zeros((10))
        target[label]= 1.0
        # 0~255의 이미지를 0~1로 정규화
        image_values = torch.FloatTensor(self.data_df.iloc[index,1:].values)
        # 레이블 ,이미지, 데이터 텐서, 목표 텐서 반환
        return label, image_values, target
    def plot_image(self,index):
        img = self.data_df.iloc[index,1:].values.reshape(28,28)
        plt.title("label = "+str(self.data_df.iloc[index,0]))
        plt.imshow(img, interpolation='none',cmap='Blues')
        plt.show()
        pass
    pass
mnist_dataset = MnistDataset('D:\study_data\_data\\test108\mnist_train.csv')
print(mnist_dataset.plot_image(9))        
        
