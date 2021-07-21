!pip install -q tensorflow-text
!pip install -q tf-models-official
!pip install -q -U keras-tuner

import os

import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization
import kerastuner as kt

import matplotlib.pyplot as plt

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
seed = 42

test_ds = tf.keras.preprocessing.text_dataset_from_directory(
          'dataset/test',
          batch_size=batch_size)

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
               'dataset/train',
               batch_size=batch_size,
               validation_split=0.2,
               subset='training',
               seed=seed)

class_names = raw_train_ds.class_names
train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

val_ds = tf.keras.preprocessing.text_dataset_from_directory(
         'dataset/train',
         batch_size=batch_size,
         validation_split=0.2,
         subset='validation',
         seed=seed)

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# model building for hyperparameter selection
def model_builder(hp):

  # neural network structure
  text_input = tf.keras.layers.Input(shape=(), 
                                     dtype=tf.string, 
                                     name='text')

  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, 
                                       name='preprocessing')

  encoder_inputs = preprocessing_layer(text_input)

  encoder = hub.KerasLayer(tfhub_handle_encoder, 
                           trainable=True, 
                           name='BERT_encoder')

  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)

  model = tf.keras.Model(text_input, net)

  # range for selecting the optimal value of the teaching step
  hp_learning_rate = hp.Choice('learning_rate', 
                                values=[1e-1, 5e-2, 1e-2, 
                                        5e-3, 1e-3, 5e-4, 
                                        1e-4, 5e-5, 1e-5, 
                                        5e-6, 1e-6])

  # loss function and metric
  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  metrics = tf.metrics.BinaryAccuracy()

  # setting parameters for training
  epochs = 5
  steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
  num_train_steps = steps_per_epoch * epochs
  num_warmup_steps = int(0.1 * num_train_steps)


  # optimizer creation
  init_lr = 3e-5
  optimizer = optimization.create_optimizer(init_lr=hp_learning_rate,
                                            num_train_steps=num_train_steps,
                                            num_warmup_steps=num_warmup_steps,
                                            optimizer_type='adamw')
  # model compilation
  model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

  return model


# hyperparameter selection
tuner = kt.Hyperband(model_builder,
                     objective='val_binary_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# the output of the best accuracy on validation for the current iteration 
# of parameter selection and the best accuracy on validation for all time 
tuner.search(train_ds, validation_data=val_ds, epochs=50, callbacks=[stop_early])

# maintaining the best hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

# building a model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)

history = model.fit(train_ds, validation_data=val_ds, epochs=2)

loss, accuracy = model.evaluate(test_ds)

print("loss: ", loss)
print("accuracy: ", accuracy)

history_dict = history.history
history_dict.keys()

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
