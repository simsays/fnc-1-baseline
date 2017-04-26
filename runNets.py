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

outfile = 'preProcessed.npz'

npzfile = np.load(outfile)
x_train = npzfile['x_train']
y_train = npzfile['y_train']
x_test = npzfile['x_val']
y_test = npzfile['y_val']
word_index = npzfile['word_index'].item()
embedding_matrix = npzfile['embedding_matrix']

CNN_FILTER_1 = 200
CNN_FILTER_2 = 100
KERNEL_SIZE = 3
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 1000

print npzfile.files
#x_train = np.reshape(x_train, (len(x_train),  28, 28, 1))
#x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

'''
# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(4, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_test, y_test))
'''




#x_train.reshape(x_train.shape + (1,))

embedding_layer = Embedding(len(word_index.keys()) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=False,
                            input_length=4894)



model = Sequential()
model.add(embedding_layer)
#y_train.reshape(y_train.shape + (1,))
#model.add(Flatten())
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
model.fit(x_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
