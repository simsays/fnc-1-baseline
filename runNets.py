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
x_train = npzfile['x_train']
y_train = npzfile['y_train']
x_val = npzfile['x_val']
y_val = npzfile['y_val']
word_index = npzfile['word_index'].item()
embedding_matrix = npzfile['embedding_matrix']
EMBEDDING_DIM = int(npzfile['embedding_dim'])
MAX_SEQUENCE_LENGTH = int(npzfile['max_seq'])

# Pre-training embedding layer
embedding_layer = Embedding(len(word_index.keys()) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=False,
                            input_length=MAX_SEQUENCE_LENGTH)


# Conv layer params
CNN_FILTER_1 = 80
CNN_FILTER_2 = 40
KERNEL_SIZE = 3

# Net structure
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(CNN_FILTER_1, KERNEL_SIZE, activation='relu'))
#model.add(Conv1D(CNN_FILTER_2, KERNEL_SIZE, activation='relu'))
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
