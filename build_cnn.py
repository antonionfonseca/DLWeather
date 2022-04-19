# Import Libraries
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import callbacks

# Initialising the CNN
classifier = Sequential()

classifier.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(250, 250, 3), padding='same', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(filters=32, kernel_size=(3, 3),  padding='same', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(filters=64, kernel_size=(3, 3),  padding='same', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(filters=64, kernel_size=(3, 3),  padding='same', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(filters=128, kernel_size=(3, 3),  padding='same', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(filters=128, kernel_size=(3, 3),  padding='same', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(GlobalAveragePooling2D())
classifier.add(Dense(512, activation="relu"))
classifier.add(Dropout(0.5))
classifier.add(Dense(128, activation="relu"))
classifier.add(Dropout(0.5))
classifier.add(Dense(4, activation="softmax"))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callback´s

lr = callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=5, min_lr=1e-30, cooldown=3, verbose=1)

file_path = 'weather/modelo'
cp = callbacks.ModelCheckpoint(file_path, monitor='accuracy', verbose=1, save_best_only=True, mode='auto')
es = callbacks.EarlyStopping(monitor='accuracy', mode='min', verbose=1, patience=15)

callbacks = [lr, es, cp]

# Summary
classifier.summary()
