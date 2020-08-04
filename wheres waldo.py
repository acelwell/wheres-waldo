import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

import os

# pull images from a directory and turn it into a directory
def loadImages(path, imgHeight, imgWidth, batchSize):
    return tf.keras.preprocessing.image_dataset_from_directory(
        path,
        labels='inferred',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(imgHeight, imgWidth),
        batch_size=batchSize
    )

#build a model
def buildModel(imgHeight, activation, layers, numClasses):
    return tf.keras.Sequential([
        keras.layers.experimental.preprocessing.Rescaling(1./255),
        keras.layers.Conv2D(imgHeight, 3, activation=activation),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(imgHeight, 3, activation=activation),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(imgHeight, 3, activation=activation),
        keras.layers.MaxPool2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(layers, activation=activation),
        keras.layers.Dense(numClasses)
    ])

def wheresWaldo():
    print("lets find waldo")


    # pull local images and turn it into a datset to be used by tensorflow, all images are 64 * 64 px
    waldo_images_64 = loadImages('./Hey-Waldo-master/sixfour', 64, 64, 32)
    validation_64 = loadImages('./Hey-Waldo-master/sixfour', 64, 64, 32)

    # same as above but 128 * 128 px
    waldo_images_128 = loadImages('./Hey-Waldo-master/128', 128, 128, 32)
    validation_128 = loadImages('./Hey-Waldo-master/128', 128, 128, 32)

    # same but 256 * 256 px
    waldo_images_256 = loadImages('./Hey-Waldo-master/256', 256, 256, 32)
    validation_256 = loadImages('./Hey-Waldo-master/256', 256, 256, 32)

    # classes are pulled from small images but all classes are the same
    class_names = waldo_images_64.class_names

    print(class_names)

    # plot figures to screen to see the first few images
    # plt.figure(figsize=(10, 10))
    # for images, labels in waldo_images.take(1):
    #     print(labels)
        # for i in range(9):
        #     # if(labels[i] == 1):
        #         ax = plt.subplot(3, 3, i + 1)
        #         plt.imshow(images[i].numpy().astype("uint8"))
        #         plt.title(class_names[labels[i]])
        #         plt.axis("off")

    # plt.show()


    # normalize data so its easier on the nn
    # normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

    # cahce images in mem so we dont have to keep fetching them
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    waldo_images_64 = waldo_images_64.cache().prefetch(buffer_size=AUTOTUNE)
    validation_64 = validation_64.cache().prefetch(buffer_size=AUTOTUNE)

    waldo_images_128 = waldo_images_128.cache().prefetch(buffer_size=AUTOTUNE)
    validation_128 = validation_128.cache().prefetch(buffer_size=AUTOTUNE)

    waldo_images_256 = waldo_images_256.cache().prefetch(buffer_size=AUTOTUNE)
    validation_256 = validation_256.cache().prefetch(buffer_size=AUTOTUNE)



    num_classes = 2

    # model build for small images
    model_64 = buildModel(64, 'relu', 64, 2)

    # build for medium images
    model_128 = buildModel(128, 'relu', 128, 2)

    # build for large images
    model_256 = buildModel(256, 'relu', 128, 2)


    print("############################# waldo 64 #####################################")
    model_64.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model_64.fit(
        waldo_images_64,
        validation_data=validation_64,
        epochs=3
    )

    # print("############################# waldo 128 #####################################")
    # model_128.compile(
    #     optimizer='adam',
    #     loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     metrics=['accuracy']
    # )
    #
    # model_128.fit(
    #     waldo_images_128,
    #     validation_data=validation_128,
    #     epochs=3
    # )
    #
    #
    # print("############################# waldo 256 #####################################")
    # model_256.compile(
    #     optimizer='adam',
    #     loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     metrics=['accuracy']
    # )
    #
    # model_256.fit(
    #     waldo_images_256,
    #     validation_data=validation_256,
    #     epochs=3
    # )





    model_64.evaluate(validation_64, verbose=2)
    # model_128.evaluate(validation_128, verbose=2)
    # model_256.evaluate(validation_256, verbose=2)

    # probability_model_64 = tf.keras.Sequential([model_64, tf.keras.layers.Softmax()])
    # prediction_64 = probability_model_64.predict(waldo_images_64)
    #
    # probability_model_128 = tf.keras.Sequential([model_128, tf.keras.layers.Softmax()])
    # prediction_128 = probability_model_128.predict(waldo_images_128)
    #
    # probability_model_256 = tf.keras.Sequential([model_256, tf.keras.layers.Softmax()])
    # prediction_256 = probability_model_256.predict(waldo_images_256)


def main():
    print("hello world")
    print(tf.__version__)

if __name__ == '__main__':
    # main()
    wheresWaldo()
    print("finished")