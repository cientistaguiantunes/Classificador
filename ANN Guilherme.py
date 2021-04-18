import os
import keras
import numpy
import pandas as pd
from sklearn import svm
import cv2
import tensorflow as tf
from keras.models import model_from_json
from sklearn.multiclass import OneVsRestClassifier
from keras.models import Model
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras import regularizers, datasets
from keras.models import Sequential
from keras import backend as K

K.set_learning_phase(1)
from keras.optimizers import SGD, RMSprop

from numpy.random import seed
import keras
# from tensorflow import set_random_seed
from importlib import reload

reload(keras.models)
import matplotlib.pyplot as plt
import matplotlib.style as sty

import numpy as np
import xml.etree.ElementTree as etree

# y = pd.read_csv('/Users/tecnicaicp/Documents/PlantCLEF2015/FOLHA/yLEAF_vgg.csv', header=None)
# y = y.to_numpy()
# y = np.ravel(y)
# print(y.shape)

# Configurações Arbritarias escolhidas para treinamento
# MUDAR CONFORME N ESPECIES
# batch_size = 115
num_classes = 351
epochs = 10

size_to_resize = [32, 32, 3]
rotulo_teste = []
rotulo_treino = []


def Model_ANN(i):
    trainX, trainY, testX, testY = load_clef_database()

    trainX = trainX.reshape(trainX.shape[0], 32, 32, 3)
    testX = testX.reshape(testX.shape[0], 32, 32, 3)

    testY = np.array(testY)

    print("Shape de treinamento")
    print(trainX.shape)
    print(trainY.shape)
    print("Shape de teste")
    print(testX.shape)
    print(testY.shape)

    np.save("/Users/tecnicaicp/Documents/PlantCLEF2015/FOLHA/trainXLeaf.npy", trainX)
    np.save("/Users/tecnicaicp/Documents/PlantCLEF2015/FOLHA/trainYLeaf.npy", trainY)
    np.save("/Users/tecnicaicp/Documents/PlantCLEF2015/FOLHA/testXLeaf.npy", testX)
    np.save("/Users/tecnicaicp/Documents/PlantCLEF2015/FOLHA/testYLeaf.npy", testY)

    # label

    # testX = np_utils.to_categorical(testX.shape[0],352)
    # testY = np_utils.to_categorical(testY.shape[0],352)

    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(28, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     input_shape=trainX.shape[1:]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(28, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=["accuracy"])

    model.summary()

    # model.fit para executar treinamento
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

    results = model.fit(trainX, trainY,
                        batch_size=64,
                        epochs=epochs,
                        verbose=2,
                        validation_data=(testX, testY),
                        # validation_split = 0.3
                        )

    # Realiza avaliação final da rede
    score = model.evaluate(trainX, trainY, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Plota o grafico do histórico de evolução da taxa de acertos da rede
    sty.use('seaborn-whitegrid')
    plt.plot(results.history["accuracy"], 'k--')
    plt.plot(results.history["val_accuracy"], 'k')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    classes = model.predict_classes(testX, batch_size=4)
    '''errado = 0
    certo = 0
    for classe in classes:
        resultado = test_rotulo[classe]
        #index = test_rotulo.index(resultado)
        if resultado == 0:
            print("Errado:" + str(classe))
            errado += 1
        else:
            print("Certo:" + str(classe))
            certo += 1
    print("Total errados:"+ str(errado))
    print("Total certos:" + str(certo))'''

    print(classes)


def load_clef_database():
    img_data_list = []

    # MUDAR DIRETORIO
    dataset_dir = "/Users/tecnicaicp/Documents/PlantCLEF2015/"
    root = os.path.join(dataset_dir, 'train')
    filenames = []  # files
    class_species = []
    class_species_unique = []
    class_species_unique_id = []
    class_familys = []
    class_geni = []
    class_ids = []
    class_contents = []
    metadata = []  # xml dat
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

    # assert len(filenames) == len(metadata)

    for i in range(len(metadata)):
        class_data = etree.parse(metadata[i])
        class_family = class_data.findtext('Family')
        class_genus = class_data.findtext('Genus')
        class_spec = class_data.findtext('Species')
        class_id = class_data.findtext('ClassId')
        class_content = class_data.findtext('Content')
        if class_content == 'LeafScan':
            # if class_spec in y:
            # print(class_content)
            leafscans.append(metadata[i])
            leafscansfiles.append(filenames[i])
            if class_spec not in class_species_unique:
                class_species_unique.append(class_spec)
            class_geni.append(class_genus)
            class_familys.append(class_family)
            class_ids.append(class_id)
            if class_id not in class_species_unique_id:
                class_species_unique_id.append(class_id)
            class_contents.append(class_content)
    print(len(leafscansfiles))

    for leaf in leafscansfiles:
        print(leaf)
        input_img = cv2.imread(leaf)
        # cv2.imshow("Original", input_img)
        # cv2.waitKey(0)
        input_img = cv2.resize(input_img, (size_to_resize[0], size_to_resize[1]))
        input_img = np.reshape(input_img, (size_to_resize[0], size_to_resize[1], size_to_resize[2]))
        img_data_list.append(input_img)
    img_data = np.array(img_data_list)

    # normalização
    img_data = img_data.astype('float32') / 255.0

    # alterar conforme o numero de imagens de cada categoria(holdout de treino e teste)
    # n_train = 1335
    # n_train = 8822

    trainX = img_data

    trainY = [] # * len(trainX)

    for i in range(len(class_ids)):
        labelX = [0] # * (len(class_species_unique_id) - 1)
        # labelY = [0] * len(class_species_unique_id)

        # if(i < len(trainX)):
        index = class_species_unique_id.index(class_ids[i])
        labelX.insert(index, 1)
        trainY.insert(i, labelX)
        # else:
        #   index = class_species_unique_id.index(class_ids[i])
        #  labelY.insert(index, 1)
        #  testY.insert(i, labelY)
    # print(testX[0])
    # print(testX[10])

    for classe in (class_species_unique_id):
        print("Class index:" + str(class_species_unique_id.index(classe)) + "Class id::" + str(classe))

    trainY = np.array(trainY)

    testX, testY = load_clef_database_test(class_species_unique_id)

    return trainX, trainY, testX, testY


def load_clef_database_test(class_species_unique_id):
    class_species_unique_id_aux = class_species_unique_id
    print(len(class_species_unique_id_aux))

    img_data_list = []
    # MUDAR DIRETORIO
    # dataset_dir = "D:/Projeto/PlantCLEF2015TestDataWithAnnotations/"
    dataset_dir = "/Users/tecnicaicp/Documents/PlantCLEF2015/PlantCLEF2015TestDataWithAnnotations"
    # root = os.path.join(dataset_dir, 'original')
    filenames = []  # files
    class_species = []
    class_species_unique = []
    class_species_unique_id = []
    class_familys = []
    class_geni = []
    class_ids = []
    class_contents = []
    metadata = []  # xml dat
    leafscans = []
    leafscansfiles = []

    for file in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, file)
        if file.endswith('.xml'):
            metadata.append(path)
        elif file.endswith('.jpg'):
            filenames.append(path)

    filenames = sorted(filenames)
    metadata = sorted(metadata)

    # assert len(filenames) == len(metadata)

    for i in range(len(metadata)):
        class_data = etree.parse(metadata[i])
        class_family = class_data.findtext('Family')
        class_genus = class_data.findtext('Genus')
        class_spec = class_data.findtext('Species')
        class_id = class_data.findtext('ClassId')
        class_content = class_data.findtext('Content')
        if class_content == 'LeafScan':
            # if class_spec in y:
            # print(class_content)
            leafscans.append(metadata[i])
            leafscansfiles.append(filenames[i])
            if class_spec not in class_species_unique:
                class_species_unique.append(class_spec)
            class_geni.append(class_genus)
            class_familys.append(class_family)
            class_ids.append(class_id)
            if class_id not in class_species_unique_id:
                class_species_unique_id.append(class_id)
            class_contents.append(class_content)
    print(len(leafscansfiles))

    for leaf in leafscansfiles:
        print(leaf)
        input_img = cv2.imread(leaf)
        # cv2.imshow("Original", input_img)
        # cv2.waitKey(0)
        input_img = cv2.resize(input_img, (size_to_resize[0], size_to_resize[1]))
        input_img = np.reshape(input_img, (size_to_resize[0], size_to_resize[1], size_to_resize[2]))
        img_data_list.append(input_img)
    img_data = np.array(img_data_list)

    # normalização
    img_data = img_data.astype('float32') / 255.0

    test_imgs = img_data

    for i in range(len(class_ids)):
        label_test = [0] * (len(class_species_unique_id_aux) - 1)
        index = class_species_unique_id_aux.index(class_ids[i])
        label_test.insert(index, 1)
        rotulo_teste.insert(i, label_test)
    # print(testX[0])
    # print(testX[10])

    for rotulo in (rotulo_teste):
        print("Class id::" + str(rotulo))

    # test_img = np.array(test_imgs)
    # test_rotulo = np.array(rotulo_teste)

    return test_imgs, rotulo_teste


if __name__ == "__main__":
    Model_ANN(35)
    print("seed=========", 35)
