from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
import cv2

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

def readScaledImage(fileName):
    img = cv2.imread(fileName)
    return cv2.resize(img, (224, 224))

    height, width, count = img.shape

    if (height > width):
        img = cv2.resize(img, ((int)(width * 224. / height), 224))
        height, width, count = img.shape
        left = (int)((224 - width)/2)
        return cv2.copyMakeBorder(img, 0, 0, left, 224 - width - left, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    else:
        img = cv2.resize(img, (224, (int)(height * 224. / width)))
        height, width, count = img.shape
        top = (int)((224 - height) / 2)
        return cv2.copyMakeBorder(img, top, 224 - height - top, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

def createNormalizedImageMask(im):
    im = im.astype(np.float32)
    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68
    im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)

    return im

def displayImageWithLabel(image, fileName, label):
    replicate = cv2.copyMakeBorder(image, 0, 30, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    cv2.putText(replicate, label, (0, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow(fileName, replicate)


names = []
with open('/Users/pnegadailov/Data/synset_words.txt') as f:
    for line in f:
        names.append(line[line.index(" ") : len(line)-1])

# Test pretrained model
model = VGG_16('/Users/pnegadailov/Data/vgg16_weights.h5')
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(optimizer=sgd, loss='categorical_crossentropy')

imageDir = '/Users/pnegadailov/Data/images/'

import os
for file in os.listdir(imageDir):
    im = readScaledImage(imageDir + file)
    out = model.predict(createNormalizedImageMask(im))
    print(file + ' -> ' + ' '*(20-len(file)) + names[np.argmax(out)])

    displayImageWithLabel(im, imageDir + file, names[np.argmax(out)])


cv2.waitKey()