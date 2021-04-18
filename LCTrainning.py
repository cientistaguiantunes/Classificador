import cv2
import numpy as np
import pandas as pd
import re
import math
from skimage import feature
from skimage.feature import hog
from imutils import paths
import os, glob
from sklearn.decomposition import PCA
from lxml import etree
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Input
from keras.models import model_from_json
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

#from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras import regularizers, datasets
from keras.models import Model

from keras import backend as K
K.set_learning_phase(1)


import matplotlib.pyplot as plt
import matplotlib.style as sty

from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

import os
import random as rn
import tensorflow as tf
import seaborn as sns
from lxml import etree


#Load Inception_v3 pretrained on ImageNet dataset
#model = InceptionV3(include_top=False, weights='imagenet', pooling='avg', input_tensor=Input(shape=(144, 96, 3)))
# load the model
model = keras.applications.VGG16(input_shape=(144, 96, 3), include_top = False, weights= 'imagenet')
# load json and create model
#json_file = open('D:/Projeto/autoencoder_diego/diego_cae/35.json', 'r')
#json_file = open('C:/diego_cae/35.json', 'r')
#model = json_file.read()
#json_file.close()
#model = model_from_json(model)
# load weights into new model
#model.load_weights("D:/Projeto/autoencoder_diego/diego_cae/35.h5")
print("Loaded model from disk")

#dataset_dir = "/Users/tecnicaicp/Documents/MESTRADO/"
dataset_dir = "/Users/tecnicaicp/Documents/PlantCLEF2015/"
root = os.path.join(dataset_dir, 'PlantCLEF2015TestDataWithAnnotations')
#root = os.path.join(dataset_dir, 'test')
#root = os.path.join(dataset_dir, 'PlantCLEF2016Test')
filenames = []	# files
class_species = []
class_familys = []
class_geni = []
class_ids = []
class_contents = []
metadata = [] # xml dat
leafscans = []
leafscansfiles = []

for file in os.listdir(root):
    path = os.path.join(root, file)
    if file.endswith('.xml'):
        metadata.append(path)
    elif file.endswith('.jpg'):
        filenames.append(path)

filenames = sorted(filenames)
metadata = sorted(metadata)

#assert len(filenames) == len(metadata)

for i in range(len(metadata)):
    class_data = etree.parse(metadata[i])
    class_family = class_data.findtext('Family')
    class_genus = class_data.findtext('Genus')
    class_spec = class_data.findtext('Species')
    class_id = class_data.findtext('ClassId')
    class_content = class_data.findtext('Content')
    if class_content == 'Flower':      #LeafScan - Fruit - Flower - Entire - Leaf - Stem - Branch
        #print(class_content)
        leafscans.append(metadata[i])
        leafscansfiles.append(filenames[i])
        class_species.append(class_spec)
        class_geni.append(class_genus)
        class_familys.append(class_family)
        class_ids.append(class_id)
        class_contents.append(class_content)
# general path
X = []
X_deep = []
y = []
print(len(leafscans))


# Feature extraction
for i in range(len(leafscans)):
    #if class_contents[i] is 'LeafScan':
        #for name in glob.glob(os.path.join(path,"*.jpg")):
        #for name in glob.glob(path+class_names+'/*.jpg'):
        name = leafscansfiles[i]
        #print(class_contents[i])

        imagem = cv2.imread(name)
        imagem = cv2.resize(imagem, (144, 96))
        #imagem = cv2.resize(imagem, (299, 299))
        #print(imagem.shape)
        altura, largura, _ = imagem.shape

     # Convert the image to RGB and Gray

        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        cinza = cv2.GaussianBlur(cinza, (21, 21), 0)

        rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

        #cv2.imshow('image', cinza)
        # Color Histograms
        r_histograma = cv2.calcHist([rgb], [0], None, [256], [0, 256]) / (altura * largura)
        g_histograma = cv2.calcHist([rgb], [1], None, [256], [0, 256]) / (altura * largura)
        b_histograma = cv2.calcHist([rgb], [2], None, [256], [0, 256]) / (altura * largura)

        # Local Binary Pattern (LBP)
        lbp = feature.local_binary_pattern(cinza, 59, 1, method="uniform")
        #lbp = feature.local_binary_pattern(rgb, 59, 1, method="uniform")
        (lbp_histograma, _) = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
        lbp_histograma = lbp_histograma.astype("float")
        lbp_histograma /= (lbp_histograma.sum())

        # Hog (Hostogram of Gradient - Direction)
        #hg = hog(cinza, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(8, 8), block_norm='L2-Hys')  #

        # Concatenate the handcrafted feature sets
        X_image = [lbp_histograma, r_histograma, g_histograma, b_histograma]
        X_image_aux = []
        for aux in X_image:
            X_image_aux = np.append(X_image_aux, np.ravel(aux))

        X_image = [i for i in X_image_aux]
        #y_class = [i for class_species[i] in range 0,100]
        print(name)
        #print(class_contents[i])
        y.append(class_species[i])
        X.append(X_image)

        # Extract deep features using InceptionV3 pretrained model   a partir daqui

        img = cv2.resize(imagem, (96, 144))
        print(img.shape)
        #img = tf.convert_to_tensor(img[:3])
        #img = np.reshape(img, (144, 96, 3))
        xd = image.img_to_array(img)
        xd = np.expand_dims(xd, axis=0)
        print(xd.shape)
        xd = preprocess_input(xd)
        deep_features = model.predict(xd)

        #print(deep_features.shape)
        print(i)
        X_image_aux = []
        for aux in deep_features:
            X_image_aux = np.append(X_image_aux, np.ravel(aux))
            deep_features = [i for i in X_image_aux]
            X_deep.append(deep_features)


# Saving the extracted features (handcrafted) in a csv file
df = pd.DataFrame(X)
df.to_csv('/Users/tecnicaicp/Documents/PlantCLEF2015/X.csv', header=False, index=False)

# Saving the extracted features (deep) in a csv file
#df = pd.DataFrame(X_deep)
#df.to_csv('/Users/tecnicaicp/Documents/PlantCLEF2015/X_deepFLOWER_vgg.csv', header=False, index=False)

# Saving the classes in a csv file
df_class = pd.DataFrame(y)
df_class.to_csv('/Users/tecnicaicp/Documents/PlantCLEF2015/y.csv', header=False, index=False)

