import numpy as np

# LSTM for sequence classification in the IMDB dataset
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers import LSTM
from keras.layers import Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical

# Preprocessed data loading
outfile = 'preProcessed.npz'
npzfile = np.load(outfile)
headline_train = npzfile['headline_train']
headline_val = npzfile['headline_val']
article_train = npzfile['article_train']
article_val = npzfile['article_val']
y_train = npzfile['y_train']
y_val = npzfile['y_val']


headline_word_index = npzfile['headline_word_index'].item()
article_word_index = npzfile['artilce_word_index'].item()
headline_embedding_matrix = npzfile['headline_embedding_matrix']
article_embedding_matrix = npzfile['article_embedding_matrix']

HEADLINE_EMBEDDING_DIM = int(npzfile['headline_embedding_dim'])
ARTICLE_EMBEDDING_DIM = int(npzfile['article_embedding_dim'])
MAX_SEQUENCE_LENGTH = int(npzfile['max_seq'])




# Conv layer params
CNN_FILTER_1 = 80
CNN_FILTER_2 = 40
KERNEL_SIZE = 3

article_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='article_input')
# Pre-training embedding layer
article_embedding_layer = Embedding(len(word_index.keys()) + 1,
                            ARTICLE_EMBEDDING_DIM,
                            weights=[article_embedding_matrix],
                            trainable=False,
                            input_length=MAX_SEQUENCE_LENGTH)

# Net structure
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(CNN_FILTER_1, KERNEL_SIZE, activation='relu'))
model.add(Flatten())

'''
model.add(MaxPooling1D(pool_length=2))
model.add(Dropout(0.2))
model.add(Conv1D(CNN_FILTER_1, KERNEL_SIZE, activation='relu'))
model.add(Flatten())
#model.add(Conv1D(CNN_FILTER_2, KERNEL_SIZE, activation='relu'))

#model.add(LSTM(100))
'''
model.add(Dense(4, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
print('Training model.')
model.fit(x_train, y_train, epochs=3, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
