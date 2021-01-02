# %% Libraries
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
# %% Train on GPU Setup
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
    
# %% Useful Functions    
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

# %% Model
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
    opt = tf.keras.optimizers.RMSprop(lr = 0.001)
    model.compile(optimizer=opt , loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# %% 
x_train, y_train, x_test, y_test = load_dataset()
x_train, x_test = prep_pixels(x_train, x_test)

model = create_model()

datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.13, height_shift_range=0.13, horizontal_flip=True)
 
datagen.fit(x_train)

iterator = datagen.flow(x_train, y_train, batch_size = 64)

#; Not the best scheduler but you get the idea.
def lr_schedule(epoch):
    lr = 0.001
    if epoch > 24:
        lr = 0.0008
    if epoch > 32:
        lr = 0.00065
    if epoch > 48:
        lr = 0.00058
    if epoch > 57:
        lr = 0.00052    
    if epoch > 64:
        lr = 0.00045
    if epoch > 77:
        lr = 0.00040
    if epoch > 92:
        lr = 0.00035
    if epoch > 110:
        lr = 0.00030
    if epoch > 128:
        lr = 0.00025        
    if epoch > 156:
        lr = 0.00021
    if epoch > 180:
        lr = 0.00016
    if epoch > 216:
        lr = 0.00009                
    return lr

lr_cb = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose = 1)

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
conv2d_3 (Conv2D)            (None, 8, 8, 64)          36928     
_________________________________________________________________
dropout (Dropout)            (None, 8, 8, 64)          0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 8, 8, 64)          36928     
_________________________________________________________________
batch_normalization_3 (Batch (None, 8, 8, 64)          256       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 64)          0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 4, 128)         73856     
_________________________________________________________________
batch_normalization_4 (Batch (None, 4, 4, 128)         512       
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 4, 4, 128)         147584    
_________________________________________________________________
dropout_1 (Dropout)          (None, 4, 4, 128)         0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 2, 2, 128)         0         
_________________________________________________________________
flatten (Flatten)            (None, 512)               0         
_________________________________________________________________
dense (Dense)                (None, 128)               65664     
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
batch_normalization_5 (Batch (None, 128)               512       
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 392,682
Trainable params: 391,786
Non-trainable params: 896
_________________________________________________________________
"""
tf.keras.utils.plot_model(model, 'model-rmsprop.png')

# %% Fit the Data
from sklearn.model_selection import train_test_split as tts
X_val, X_test, Y_val, Y_test = tts(x_test, y_test, train_size = 0.7)

history = model.fit(iterator, epochs=784, validation_data=(X_val, Y_val), verbose=1,callbacks= [es_cb, md_cp,lr_cb])
"""
Epoch 00289: LearningRateScheduler reducing learning rate to 9e-05.
782/782 [==============================] - 35s 45ms/step - loss: 0.3662 - accuracy: 0.9084 - val_loss: 0.4910 - val_accuracy: 0.8734
"""

acc = model.evaluate(X_test, Y_test, verbose=1)
"""
94/94 [==============================] - 1s 8ms/step - loss: 0.5075 - accuracy: 0.8613
"""

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