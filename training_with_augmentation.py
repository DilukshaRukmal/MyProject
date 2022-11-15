from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

IMAGE_SIZE = 224
CHANNELS = 3

from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = "Fold1\Fold1\Fold1/Train" # path to training set
training_datagen = ImageDataGenerator(                # here apply some real time augmentations to images
      rescale = 1./255,                               #sclale the size of image between 0-1
	  rotation_range=40,                              # rotated in range 0-40
      width_shift_range=0.2,                          #width shift
      height_shift_range=0.2,                         #height shift
      shear_range=0.2,
      zoom_range=0.2,                                 #zoom image 
      horizontal_flip=True,
      fill_mode='nearest')                            #fill the rest with nearest pixel

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size = (IMAGE_SIZE,IMAGE_SIZE),                    #image resized to 224x224px
    class_mode = 'binary',                                   #2D one hot encoded
    batch_size=10
)


VALIDATION_DIR = "Fold1\Fold1\Fold1/Val"  #path to validation set
validation_datagen = ImageDataGenerator(rescale = 1./255)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size = (IMAGE_SIZE,IMAGE_SIZE),
    class_mode = 'binary'
)

TEST_DIR = "Fold1\Fold1\Fold1/Test"
test_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        horizontal_flip=True)

test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMAGE_SIZE,IMAGE_SIZE),
        class_mode="binary"
)

print(test_generator.class_indices)
class_names = list(train_generator.class_indices.keys())

input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3

model = tf.keras.models.Sequential([
    keras.layers.InputLayer(input_shape=input_shape),
    keras.layers.Conv2D(32, kernel_size = (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    steps_per_epoch=47,
    batch_size=32,
    validation_data=validation_generator,
    validation_steps=6,
    verbose=1,
    epochs=20,
)


loss = history.history['loss']
accuracy= history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']


EPOCHS = 20

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), accuracy, label='Training Accuracy')
plt.plot(range(EPOCHS), val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


for image_batch, label_batch in test_generator:
    first_image = image_batch[0]
    first_label = int(labels_batch[0])
    
    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:",class_names[first_label])
    
    batch_prediction = model.predict(images_batch)
    print("predicted label:",class_names[np.argmax(batch_prediction[0])])
    
    break