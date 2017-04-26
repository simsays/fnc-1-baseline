import numpy as np

# LSTM for sequence classification in the IMDB dataset
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Conv1D
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

print npzfile.files
print len(embedding_matrix)
print len(embedding_matrix[0])

y_train = to_categorical(y_train)

embedding_layer = Embedding(len(word_index.keys()) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=False,
                            input_length=4894,
                            return_sequences=True)

# fix random seed for reproducibility
np.random.seed(7)

model = Sequential()
model.add(embedding_layer)
#model.add(Conv1D(CNN_FILTER_1, KERNEL_SIZE, activation='relu'))
#model.add(Conv1D(CNN_FILTER_2, KERNEL_SIZE, activation='relu'))

#model.add(LSTM(100))
#model.add(Dense(1, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))