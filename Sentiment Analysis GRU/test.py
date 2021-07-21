import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

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

maxWordsCount = 100000
tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»', lower=True, split=' ', char_level=False)
tokenizer.fit_on_texts(texts)

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


max_test_text_len = 30
test_data = tokenizer.texts_to_sequences(test_text)
test_data_pad = pad_sequences(test_data, maxlen=max_test_text_len)
print(test_data_pad.shape)

X = test_data_pad
Y = np.array([[1, 0]] * count_true)

for i in range(count_true):
  if test_true[i] == 1:
    Y[[i]] = [[0, 1]]

indeces = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
X = X[indeces]
Y = Y[indeces]

model = tf.keras.models.load_model('model')
model.evaluate(X, Y)

examples = [
  "You stupid",
  "I want to die",
  "Today is a good day",
  "I'm lost",
  "You looks really nice",
  "Good morning"]
data = tokenizer.texts_to_sequences(examples)
data_pad = pad_sequences(data, maxlen=max_test_text_len)

res = model.predict(data_pad)
print(res, np.argmax(res), sep='\n')