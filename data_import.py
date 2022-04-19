# Import Libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# This helps to expose the model to more aspects of the data
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# No modifications are made in the test set
test_datagen = ImageDataGenerator(rescale=1./255)

######################################################################
batch_size = 32
val_batch_size = 32

training_set = train_datagen.flow_from_directory('weather/data_train',
                                                 target_size=(250, 259),
                                                 batch_size=batch_size,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('weather/data_test',
                                            target_size=(250, 250),
                                            batch_size=val_batch_size,
                                            class_mode='categorical',
                                            shuffle=True)
