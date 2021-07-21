import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import re

from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
import kerastuner as kt
import matplotlib.pyplot as plt

with open('dataset/positive.txt', 'r', encoding='utf-8') as f:
    texts_true = f.readlines() # positive comments
    texts_true[0] = texts_true[0].replace('\ufeff', '')

with open('dataset/negative.txt', 'r', encoding='utf-8') as f:
    texts_false = f.readlines() # negative comments
    texts_false[0] = texts_false[0].replace('\ufeff', '')

texts = texts_true + texts_false

count_true = len(texts_true) # number of positive comments
count_false = len(texts_false) # number of negative comments
total_lines = count_true + count_false # total number of comments

maxWordsCount = 100000 # maximum number of words
tokenizer = Tokenizer(num_words=maxWordsCount, 
                      filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»', 
                      lower=True, 
                      split=' ', 
                      char_level=False)
tokenizer.fit_on_texts(texts) 

max_text_len = 30 # maximum line length

# convert text to a numeric sequence
data = tokenizer.texts_to_sequences(texts)

# if the vector is shorter than the maximum string length, it is padded with zeros at the beginning, and if it is longer, it is truncated 
data_pad = pad_sequences(data, maxlen=max_text_len)

X = data_pad # training sample

# selection of desired values
Y = np.array([[1, 0]] * count_true + [[0, 1]] * count_false) 

# Mixing the sample for better learning
indeces = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
X = X[indeces]
Y = Y[indeces]

with open('dataset/test.txt', 'r', encoding='utf-8') as f:
    test_text = f.readlines() # test phrases
    test_text[0] = test_text[0].replace('\ufeff', '')

with open('dataset/test_true.txt', 'r', encoding='utf-8') as f:
    test_true = f.readlines() # desired values
    test_true[0] = test_true[0].replace('\ufeff', '')

count_test = len(test_text) # number of test phrases
count_true = len(test_true) # number of desired values

for i in range(count_true):
    test_true[i] = int(test_true[i].replace('\n', ''))

test_data = tokenizer.texts_to_sequences(test_text)
test_data_pad = pad_sequences(test_data, maxlen=max_text_len)

X_test = test_data_pad # test phrases
Y_test = np.array([[1, 0]] * count_true) # selection of desired values

for i in range(count_true):
  if test_true[i] == 1:
    Y_test[[i]] = [[0, 1]]

# mix the sample
indeces = np.random.choice(X_test.shape[0], size=X_test.shape[0], replace=False)
X_test = X_test[indeces]
Y_test = Y_test[indeces]

# model building for hyperparameter selection
def model_builder(hp):

    # number of neurons on the embedding layer
    embedding_units = hp.Int("embedding_units", 
                              min_value=32, 
                              max_value=512, 
                              step=32)
    
    # number of neurons on the LSTM layer
    lstm_units = hp.Int("lstm_units", 
                         min_value = 32, 
                         max_value=512,  
                         step=32)

    # probability of neuron disconnection on the first LSTM layer
    dropout_units_1 = hp.Float("dropout_units_1", 
                                min_value=0.1, 
                                max_value=0.5, 
                                step=0.1)

    # probability of neuron disconnection on the second LSTM layer
    dropout_units_2 = hp.Float("dropout_units_2", 
                                min_value=0.1, 
                                max_value=0.5, 
                                step=0.1)

    # model building
    model = Sequential()
    model.add(Embedding(maxWordsCount, 
                        embedding_units, 
                        input_length = max_text_len))

    model.add(LSTM(embedding_units, 
                   return_sequences=True, 
                   recurrent_dropout=dropout_units_1, 
                   recurrent_activation='sigmoid'))

    model.add(LSTM(lstm_units, 
                   return_sequences=True, 
                   recurrent_dropout=dropout_units_2, 
                   recurrent_activation='sigmoid'))

    model.add(LSTM(lstm_units, 
                   recurrent_activation='sigmoid'))

    model.add(Dense(2, activation='softmax'))

    # range for selecting the optimal value of the teaching step
    learning_rate = hp.Choice('learning_rate', 
                               values=[1e-1, 5e-2, 1e-2, 
                                       5e-3, 1e-3, 5e-4, 
                                       1e-4, 5e-5, 1e-5, 
                                       5e-6, 1e-6])

    # model compilation
    model.compile(loss='binary_crossentropy', 
                  metrics=['accuracy'], 
                  optimizer=Adam(learning_rate))

    return model

# hyperparameter selection
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3)

tuner.search(X, Y, epochs=50, validation_split=0.2)

# maintaining the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# building a model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)

history = model.fit(X, Y, epochs=1, validation_split=0.2)
loss, accuracy = model.evaluate(X, Y)

print("loss: ", loss)
print("accuracy: ", accuracy)

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()

model.save('model')
