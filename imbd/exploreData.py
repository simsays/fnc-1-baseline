# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)
print type(X_train[0])