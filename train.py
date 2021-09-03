#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import toml
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib
from models import RCNN, DenseANN, BasicCNN
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os
from mobilenet import pretrained_model


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=255):
    def eraser(input_img):
        img_h, img_w, _ = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser


class SpeakerClassifier:
    def __init__(self):
        self.model = None
        self.train_generator = None
        self.test_generator = None
        self.config = toml.load(
            pathlib.Path(
                r"/home/tharun/speaker-count/Tharun"
                r"/config.toml"))
        self.config = self.config["preprocess"]

    def configure(self, ):
        # print(tf.__version__)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    def model_loader(self, ):
        # 1 - Load the model and its pretrained weights if exists
        # classifier = cnn()
        # classifier.load('weights/cnn_DF')
        initial_learning_rate = self.config['lr']
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True)

        # self.model = RCNN(kernel_size=self.config["kernel_size"], filters=self.config["filters"])
        self.model = pretrained_model()
        #self.model = BasicCNN(kernel_size=self.config["kernel_size"], filters=self.config["filters"])
        # self.model = DenseANN(1024)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(
                               learning_rate=lr_schedule, beta_1=self.config['beta_1'],
                               beta_2=self.config['beta_2'], epsilon=1e-7), metrics=['accuracy'])

    def data_loader(self):
        # let us use ImageDataGenerator to pick data to be trained
        #curr = pathlib.Path.cwd()
        train_path = '/home/tharun/speaker-count/dataset/train'
        test_path = '/home/tharun/speaker-count/dataset/test'

        # define train data generator
        train_datagen = ImageDataGenerator(
            # featurewise_center=True,
            samplewise_std_normalization=True,
            width_shift_range=0.2,
            horizontal_flip=True,
            preprocessing_function=get_random_eraser(v_l=0, v_h=255)
        )
        self.train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(224, 224),  # All images will be resized to 256x388
            batch_size=self.config['batch_size'],
            color_mode='rgb',
            class_mode='categorical',
            shuffle=True
        )

        # define test data generator
        test_datagen = ImageDataGenerator(
            samplewise_std_normalization=True
        )
        self.test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=(224, 224),  # All images will be resized to 150x150
            batch_size=self.config['batch_size'],
            color_mode='rgb',
            class_mode='categorical',
            shuffle=True
        )

    def train(self, ):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
        self.configure()
        print("configuration is done")
        self.model_loader()
        print("Loaded the model")
        self.data_loader()
        print("DataSet loaded")
        # start training
        print("calling fit function on it")
        callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=1, mode="min")  # was 10
        history = self.model.fit(
            self.train_generator,
            steps_per_epoch=575,
            epochs=300,
            verbose=1,
            validation_data=self.test_generator,
            validation_steps=262,
            callbacks=[callback],
        )
        
        print(history.history.keys())
        #  "Accuracy"
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig("accuracy.png")
        plt.close()
        # "Loss"
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig("loss.png")
        plt.close()
        
        # Evaluate the base model
        train_score =self.model.evaluate(self.train_generator, verbose=1)
        print('Train accuracy:', train_score[1])

        valid_score =self.model.evaluate(self.test_generator, verbose=1)
        print('Validation accuracy:', valid_score[1])
        # self.model.save('newweights/Conv5.h5')


if __name__ == "__main__":
    classifier = SpeakerClassifier()
    classifier.train()