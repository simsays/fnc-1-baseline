
###########################################
# Imports
import numpy as np
from keras.models import Sequential, Model, load_model, model_from_json
from keras.layers import Dense, Input, Merge
from keras.layers import LSTM
from keras.layers import Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.optimizers import SGD


#########################################
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
article_word_index = npzfile['article_word_index'].item()
headline_embedding_matrix = npzfile['headline_embedding_matrix']
article_embedding_matrix = npzfile['article_embedding_matrix']
HEADLINE_EMBEDDING_DIM = int(npzfile['headline_embedding_dim'])
ARTICLE_EMBEDDING_DIM = int(npzfile['article_embedding_dim'])
MAX_SEQUENCE_LENGTH = int(npzfile['max_seq'])


#######################################
# Neural Network structure
#######################################

#######################################
# Article Branch
# Pre-training embedding layer
article_branch = Sequential()
article_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='article_input')
article_embedding_layer = Embedding(len(article_word_index.keys()) + 1,
                            ARTICLE_EMBEDDING_DIM,
                            weights=[article_embedding_matrix],
                            trainable=False,
                            input_length=MAX_SEQUENCE_LENGTH)
article_branch.add(article_embedding_layer)

# Conv layer params
CNN_FILTER_1 = 200
CNN_FILTER_2 = 80
CNN_FILTER_3 = 50
KERNEL_SIZE = 3

# CNN
conv1 = Conv1D(CNN_FILTER_1, KERNEL_SIZE, activation='relu', padding = 'same')
article_branch.add(conv1)

conv2 = Conv1D(CNN_FILTER_2, KERNEL_SIZE, activation='relu', padding = 'same')
article_branch.add(conv2)

dropout = Dropout(.1) # Dropout layer
article_branch.add(dropout)

conv3 = Conv1D(CNN_FILTER_3, KERNEL_SIZE, activation='relu', padding = 'same')
article_branch.add(conv3)

print(article_branch.summary())


#########################################
# Headline Branch
headline_branch = Sequential()
headline_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='headline_input')
headline_embedding_layer = Embedding(len(headline_word_index.keys()) + 1,
                            HEADLINE_EMBEDDING_DIM,
                            weights=[headline_embedding_matrix],
                            trainable=False,
                            input_length=MAX_SEQUENCE_LENGTH)
headline_branch.add(headline_embedding_layer)

print(headline_branch.summary())


########################################
# Merged Branch
model = Sequential()
model.add(Merge([article_branch, headline_branch], mode = 'concat'))

# LSTM
lstm = LSTM(20, activation='sigmoid')
model.add(lstm)

# Output Softmax Layer
output = Dense(4, activation='softmax')
model.add(output)

# Optimizer
sgd = SGD(lr=0.3, momentum=0.8, decay=.00000005, nesterov=False)

# Model compilation and training
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
print('Training model.')
model.fit([article_train, headline_train], y_train, epochs=100, batch_size=128)


######################################
# Final evaluation of the model
scores = model.evaluate([article_val, headline_val], y_val, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

######################################
# Saving Model
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


######################################
# Loading Model
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
scores = model.evaluate([article_val, headline_val], y_val, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))