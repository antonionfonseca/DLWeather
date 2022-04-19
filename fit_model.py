# Import Libraries
from data_import import training_set, test_set, batch_size, val_batch_size
from build_cnn import classifier, callbacks

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Fit the model
hist = classifier.fit(training_set,
                      steps_per_epoch=training_set.n // batch_size,
                      epochs=30,
                      validation_data=test_set,
                      validation_steps=test_set.n // val_batch_size,
                      callbacks=callbacks)
