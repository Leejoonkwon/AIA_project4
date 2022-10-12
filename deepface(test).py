import os
import numpy as np
import pandas as pd
from pathlib import Path
import zipfile
import cv2
import tqdm
import matplotlib.pyplot as plt

home = str(Path.home())
print("HOME_FOLDER is :",home) # HOME_FOLDER is : C:\Users\AIA

if not os.path.exists(home+"/.deepface"):
    os.mkdir(home+"/.deepface")
# deepface 라는 폴더가 없다면 생성하기
if not os.path.exists(home+"/.deepface/weights"):
    os.mkdir(home+"/.deepface/weigths")
# deepface라는 폴더 내에 weigths 폴더가 없다면 생성하기

import shutil
shutil.copy("../input/pretrained-models/vgg_face_weights.h5", home +"/.deepface/weigths")
shutil.copy("../input/pretrained-models/facenet_weights.h5", home +"/.deepface/weigths")
shutil.copy("../input/pretrained-models/arcface_weights.h5", home +"/.deepface/weigths")








