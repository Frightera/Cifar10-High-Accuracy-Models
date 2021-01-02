# Libraries
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

# load train and test dataset
def load_dataset():
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)
	return x_train, y_train, x_test, y_test
 
# scale pixels
def prep_pixels(train, test):
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	return train_norm, test_norm

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation = 'selu', input_shape = (32,32,3), kernel_initializer = 'he_uniform', bias_initializer ='ones', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3,3), activation = 'selu', input_shape = (32,32,3), kernel_initializer = 'he_uniform', bias_initializer ='ones', 
                               padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.MaxPooling2D(pool_size = (2,2)),

        tf.keras.layers.BatchNormalization(momentum = 0.9),

        tf.keras.layers.Conv2D(64, (3,3), activation = 'elu', kernel_initializer = 'he_uniform', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.BatchNormalization(momentum = 0.9),
        tf.keras.layers.MaxPooling2D(pool_size = (2,2)),

        tf.keras.layers.Dropout(0.24),

        tf.keras.layers.Conv2D(64, (3,3), activation = 'elu', kernel_initializer = 'he_uniform', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.BatchNormalization(momentum = 0.9),
        tf.keras.layers.MaxPooling2D(pool_size = (2,2)),

        tf.keras.layers.Conv2D(64, (3,3), activation = 'elu', kernel_initializer = 'he_uniform', padding = 'same',kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(64, (3,3), activation = 'elu', kernel_initializer = 'he_uniform', padding = 'same', kernel_regularizer = tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.BatchNormalization(momentum = 0.9),
        tf.keras.layers.MaxPooling2D(pool_size = (2,2)),

        tf.keras.layers.Conv2D(128, (3,3), activation = 'elu', kernel_initializer = 'he_uniform', padding = 'same', kernel_regularizer = tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3,3), activation = 'elu', kernel_initializer = 'he_uniform', padding = 'same',kernel_regularizer = tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.MaxPooling2D(pool_size = (2,2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation = 'elu', kernel_regularizer = tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(10, activation = 'softmax')
    ])
    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt , loss='categorical_crossentropy', metrics=['accuracy'])
    return model

x_train, y_train, x_test, y_test = load_dataset()
x_train, x_test = prep_pixels(x_train, x_test)

model = create_model()

datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.13, height_shift_range=0.13, horizontal_flip=True)
 
datagen.fit(x_train)

iterator = datagen.flow(x_train, y_train, batch_size = 64)


es_cb = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 32)
md_cp = tf.keras.callbacks.ModelCheckpoint(filepath='model.h5', save_best_only=True, verbose = 1)
 
model.summary()
"""
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 32, 32, 32)        896       
_________________________________________________________________
batch_normalization (BatchNo (None, 32, 32, 32)        128       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 32, 32)        9248      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 16, 16, 32)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 16, 16, 32)        128       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 16, 64)        18496     
_________________________________________________________________
batch_normalization_2 (Batch (None, 16, 16, 64)        256       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 64)          0         
_________________________________________________________________
dropout (Dropout)            (None, 8, 8, 64)          0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 8, 64)          36928     
_________________________________________________________________
batch_normalization_3 (Batch (None, 8, 8, 64)          256       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 64)          0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 4, 64)          36928     
_________________________________________________________________
dropout_1 (Dropout)          (None, 4, 4, 64)          0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 4, 64)          36928     
_________________________________________________________________
batch_normalization_4 (Batch (None, 4, 4, 64)          256       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 2, 2, 64)          0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 2, 2, 128)         73856     
_________________________________________________________________
batch_normalization_5 (Batch (None, 2, 2, 128)         512       
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 2, 2, 128)         147584    
_________________________________________________________________
dropout_2 (Dropout)          (None, 2, 2, 128)         0         
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 1, 1, 128)         0         
_________________________________________________________________
flatten (Flatten)            (None, 128)               0         
_________________________________________________________________
dense (Dense)                (None, 128)               16512     
_________________________________________________________________
dropout_3 (Dropout)          (None, 128)               0         
_________________________________________________________________
batch_normalization_6 (Batch (None, 128)               512       
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 380,714
Trainable params: 379,690
Non-trainable params: 1,024
_________________________________________________________________
"""

from sklearn.model_selection import train_test_split as tts
X_val, X_test, Y_val, Y_test = tts(x_test, y_test, train_size = 0.7)

history = model.fit(iterator, epochs=784, validation_data=(X_val, Y_val), verbose=1,callbacks= [es_cb, md_cp])
"""
Epoch 00141: val_loss did not improve from 0.62430
Epoch 142/784
782/782 [==============================] - 26s 33ms/step - loss: 0.7075 - accuracy: 0.8165 - 
                                            val_loss: 0.6310 - val_accuracy: 0.8446
"""

acc = model.evaluate(X_test, Y_test, verbose=1)
# 94/94 [==============================] - 0s 4ms/step - loss: 0.6280 - accuracy: 0.8447

acc      = history.history['accuracy']
val_acc  = history.history['val_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss']

epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot  ( epochs,     acc, label = 'train acc' )
plt.plot  ( epochs, val_acc, label = 'val_acc' )
plt.title ('Training and validation accuracy')
plt.legend()
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot  ( epochs,     loss, label = 'TrainLoss' )
plt.plot  ( epochs, val_loss, label = 'Val_loss' )
plt.title ('Training and validation loss'   )
plt.legend()
plt.show()



