
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

# Optimizer
sgd = SGD(lr=0.3, momentum=0.8, decay=.00000005, nesterov=False)

# Model compilation and training
loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(loaded_model.summary())
print('Training model.')
loaded_model.fit([article_train, headline_train], y_train, epochs=100, batch_size=128)


######################################
# Final evaluation of the model
scores = loaded_model.evaluate([article_val, headline_val], y_val, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

######################################
# Saving Model
# serialize model to JSON
model_json = loaded_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
loaded_model.save_weights("model.h5")
print("Saved model to disk")