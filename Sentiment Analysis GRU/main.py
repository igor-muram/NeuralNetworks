import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import re

from tensorflow.keras.layers import Dense, GRU, Input, Dropout, Embedding, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import kerastuner as kt
import matplotlib.pyplot as plt

with open('dataset/positive.txt', 'r', encoding='utf-8') as f:
    texts_true = f.readlines()
    texts_true[0] = texts_true[0].replace('\ufeff', '')

with open('dataset/negative.txt', 'r', encoding='utf-8') as f:
    texts_false = f.readlines()
    texts_false[0] = texts_false[0].replace('\ufeff', '')

texts = texts_true + texts_false
count_true = len(texts_true)
count_false = len(texts_false)
total_lines = count_true + count_false
print(count_true, count_false, total_lines)

maxWordsCount = 100000
tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»', lower=True, split=' ', char_level=False)
tokenizer.fit_on_texts(texts)

max_text_len = 30
data = tokenizer.texts_to_sequences(texts)
data_pad = pad_sequences(data, maxlen=max_text_len)
print(data_pad.shape)

X = data_pad
Y = np.array([[1, 0]] * count_true + [[0, 1]] * count_false)
print(X.shape, Y.shape)

indeces = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
X = X[indeces]
Y = Y[indeces]


with open('dataset/test.txt', 'r', encoding='utf-8') as f:
    test_text = f.readlines()
    test_text[0] = test_text[0].replace('\ufeff', '')

with open('dataset/test_true.txt', 'r', encoding='utf-8') as f:
    test_true = f.readlines()
    test_true[0] = test_true[0].replace('\ufeff', '')

count_test = len(test_text)
count_true = len(test_true)

for i in range(count_true):
    test_true[i] = int(test_true[i].replace('\n', ''))

test_data = tokenizer.texts_to_sequences(test_text)
test_data_pad = pad_sequences(test_data, maxlen=max_text_len)
print(test_data_pad.shape)

X_test = test_data_pad
Y_test = np.array([[1, 0]] * count_true)

for i in range(count_true):
  if test_true[i] == 1:
    Y_test[[i]] = [[0, 1]]

indeces = np.random.choice(X_test.shape[0], size=X_test.shape[0], replace=False)
X_test = X_test[indeces]
Y_test = Y_test[indeces]

model = Sequential()
model.add(Embedding(maxWordsCount, 128, input_length = max_text_len))
model.add(GRU(128, return_sequences=True, recurrent_dropout=0.3, recurrent_activation='sigmoid'))
model.add(GRU(32, return_sequences=True, recurrent_dropout=0.2, recurrent_activation='sigmoid'))
model.add(GRU(32, recurrent_activation='sigmoid'))
model.add(Dense(2, activation='softmax'))
    
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam(0.005))
history = model.fit(X, Y, epochs=2, validation_split=0.2)

model.evaluate(X, Y)

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

model.save('model')