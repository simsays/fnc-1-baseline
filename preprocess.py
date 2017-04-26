# Imports for tokenizing/word2vec stuff
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import os

import numpy as np

from utils.dataset import DataSet

# 10% cross-validation
VALIDATION_SPLIT = 0.1
BASE_DIR = os.getcwd()
GLOVE_DIR = BASE_DIR + '/glove.6B/'
EMBEDDING_DIM = 300

dataset = DataSet()

# texts contains the headline pre-appended to each article.
texts = []
labels = []

# The 4 stances are 'agree', 'disagree', 'unrelated', and 'discuss'

stanceDict = {'agree': 0, 'disagree': 1, 'unrelated': 2, 'discuss': 3}

for stance in dataset.stances:
	# Put outputs into labels
	labels.append(stanceDict[stance['Stance']])
	
	texts.append(stance['Headline'] + " " + dataset.articles[stance['Body ID']])



tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' %len(word_index))

data = pad_sequences(sequences)
	
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# END OF PREPROCESSING

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


outfile = "preProcessed.npz"
np.savez(outfile, x_train=x_train, y_train=y_train,\
 x_val=x_val, y_val=y_val, embedding_matrix=embedding_matrix,\
 word_index=word_index)

