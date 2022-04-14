# Part 2 --> Fitting the CNN to the images
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('weather\data_train',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('weather\data_test',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary',
                                            shuffle=False)



# Fit the model
#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#from cnn_part1 import classifier

#hist = classifier.fit(training_set,
#                      steps_per_epoch=250,
#                      epochs=5,
#                      validation_data=test_set,
#                      validation_steps=63)














