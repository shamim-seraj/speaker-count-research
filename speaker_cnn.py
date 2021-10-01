

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras import activations
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def cnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), input_shape=(100, 50, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Conv2D(16, (3, 3)))
    #model.add(layers.Activation(activations.relu))
    #model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.relu))
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Conv2D(16, (3, 3)))
    #model.add(layers.Activation(activations.relu))
    #model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    #model.add(layers.Dense(1000))
    #model.add(layers.Dropout(0.5))
    #model.add(layers.Activation(activations.relu))
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dense(64))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dense(3, activation='softmax'))
    return model


if __name__ == '__main__':

    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))
    from tensorflow.python.client import device_lib
    device_lib.list_local_devices()
    print(tf.__version__)

    train_path = '/home/shuvornb/Desktop/NIJ-AI-SMS/testdata/exported/dataset/train'
    test_path = '/home/shuvornb/Desktop/NIJ-AI-SMS/testdata/exported/dataset/test'

    # define train data generator
    train_data_generator = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    train_generator = train_data_generator.flow_from_directory(
        train_path,
        target_size=(100, 50),
        batch_size=10,
        color_mode='rgb',
        class_mode='categorical'
    )

    # define test data generator
    test_data_generator = ImageDataGenerator(
        rescale=1. / 255
    )
    test_generator = test_data_generator.flow_from_directory(
        test_path,
        target_size=(100, 50),
        batch_size=10,
        color_mode='rgb',
        class_mode='categorical',
        shuffle=True
    )

    print(train_generator.classes)
    print(train_generator.class_indices)
    print(test_generator.classes)
    print(test_generator.class_indices)

    model = cnn_model()
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=1,
        mode="auto"
    )
    
    history = model.fit(
        train_generator,
        steps_per_epoch=6000,
        epochs=100,
        validation_data=test_generator,
        validation_steps=1500,
        callbacks=[early_stopping]
    )

    # Evaluate the base model
    train_score = model.evaluate(train_generator, verbose=1)
    print('Train accuracy:', train_score[1])

    valid_score = model.evaluate(test_generator, verbose=1)
    print('Validation accuracy:', valid_score[1])

    # list all data in history
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
