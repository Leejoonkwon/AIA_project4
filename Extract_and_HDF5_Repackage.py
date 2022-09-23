import torch
import torch.nn as nn
from torch.utils.data import dataset
import h5py
import zipfile
import imageio
import os
import numpy as np
import matplotlib.pyplot as plt



# with h5py.File('D:\study_data\_data\source1.zip','r')
# 생선된 hdf5  파일의 저장 위치 및 이름 지정
hdf5_file = 'D:\study_data\_data\\test108\celeba_aligned_small.h5py'
# 해당 zip 또는 폴더 안에서 사용할 이미지의 개수
total_images = 20000

with h5py.File(hdf5_file,'w') as hf:
    count = 0
    with zipfile.ZipFile('D:\study_data\_data\\test108\source1.zip','r') as zf:
        for i in zf.namelist():
            if(i[-4:] == '.jpg'):
                #extract image
                ofile = zf.extract(i)
                img = imageio.imread(ofile)
                os.remove(ofile)
                #add image data to HDF5 file with new name
                hf.create_dataset('source1/'+str(count)+'.jpg',data=img,
                                  compression="gzip",compression_opts=9)
                count = count + 1
                if (count%1000 ==0):
                    print("image done ..",count)
                    pass
                if (count == total_images):
                    break
                pass
            pass
        pass


