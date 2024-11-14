import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 

import os
import math
import urllib.request
import zipfile

from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.layers import Activation, BatchNormalization, Input
from keras.optimizers import adam_v2, sgd_experimental
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator

from google.colab import files

# ディレクトリのパス
train_dir = "./dataset/trainData"
valid_dir = "./dataset/validData"
all_data_dir = "./tmp/Data"
source_dir = "./tmp/trainData"

# ディレクトリ下のデータ置き場所
os.makedirs("%s/dogs" %train_dir)
os.makedirs("%s/cats" %train_dir)
os.makedirs("%s/dogs" %valid_dir)
os.makedirs("%s/cats" %valid_dir)
os.makedirs("%s" %all_data_dir)
os.makedirs("%s" %source_dir)

# ファイルをアップロード
uploaded = files.upload()

# keyに対する処理
for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(name=fn, length=len(uploaded[fn])))
!mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
