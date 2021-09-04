from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.callbacks import ModelCheckpoint

#Some code for this section is gotten from https://www.tensorflow.org/tutorials/images/classification
dataset_path = "dataset"


batch_size = 16
img_size = 100;

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    
    dataset_path, validation_split=0.2, subset="training", seed=123,
    image_size=(img_size, img_size), batch_size=batch_size)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    
    dataset_path, validation_split=0.2, subset="validation", seed=123,
    image_size=(img_size, img_size), batch_size=batch_size)

categories = train_dataset.class_names


AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

number_of_categories = 2

model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_size, img_size, 3)),
    
    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(number_of_categories)
    ])


epochs = 15

model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,mode='auto')
my_model = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=[checkpoint])

plt.plot(my_model.history['loss'],'r',label='training loss')
plt.plot(my_model.history['val_loss'],label='validation loss')
plt.xlabel('# Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.plot(my_model.history['accuracy'],'r',label='training accuracy')
plt.plot(my_model.history['val_accuracy'],label='validation accuracy')
plt.xlabel('# epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()