from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# Data preprocessing
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Initializing the classifier Network
classifier = models.Sequential()
classifier.add(layers.LSTM(128, input_shape=(X_train.shape[1:]), return_sequences=True))
classifier.add(layers.Dropout(0.2))
classifier.add(layers.LSTM(128))
classifier.add(layers.Dense(64, activation='relu'))
classifier.add(layers.Dropout(0.2))
classifier.add(layers.Dense(10, activation='softmax'))

# Compiling the network
classifier.compile( loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=0.001, decay=1e-6),
              metrics=['accuracy'] )

#Fitting the data to the model
classifier.fit(X_train,
         y_train,
          epochs=3,
          validation_data=(X_test, y_test))

test_loss, test_acc = classifier.evaluate(X_test, y_test)
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))
