########################################
# Imports for tokenizing
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from utils.dataset import DataSet

#######################################
# Constants
VALIDATION_SPLIT = 0.1 # 10% validation
BASE_DIR = os.getcwd()
GLOVE_DIR = BASE_DIR + '/glove.6B/'
HEADLINE_EMBEDDING_DIM = 50 #50/100/200/300
ARTICLE_EMBEDDING_DIM = 300 #50/100/200/300
MAX_NB_WORDS = 1000
MAX_SEQUENCE_LENGTH = 1000
dataset = DataSet()


####################################
# putting data into lists
headlines = []
articles = []
stances = []

# The 4 stances are 'agree', 'disagree', 'unrelated', and 'discuss'
stanceDict = {'agree': 0, 'disagree': 1, 'unrelated': 2, 'discuss': 3}

for stance in dataset.stances:
# Put outputs into stances
    stances.append(stanceDict[stance['Stance']])
    headlines.append(stance['Headline'])
    articles.append(dataset.articles[stance['Body ID']])


#####################################
# tokenization
headline_tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
headline_tokenizer.fit_on_texts(headlines)
headline_sequences = headline_tokenizer.texts_to_sequences(headlines)

article_tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
article_tokenizer.fit_on_texts(articles)
article_sequences = article_tokenizer.texts_to_sequences(articles)

headline_word_index = headline_tokenizer.word_index
article_word_index = article_tokenizer.word_index
print('Found %s unique headline tokens.' %len(headline_word_index))
print('Found %s unique article tokens.' %len(article_word_index))


####################################
# spliting data into vectors
headline_data = pad_sequences(headline_sequences, maxlen=MAX_SEQUENCE_LENGTH)
article_data = pad_sequences(article_sequences, maxlen=MAX_SEQUENCE_LENGTH)
stances = to_categorical(np.asarray(stances))
print('Shape of headline data tensor:', headline_data.shape)
print('Shape of article data tensor:', article_data.shape)
print('Shape of stance tensor:', stances.shape)

# split the data into a training set and a validation set and shuffle it
indices = np.arange(headline_data.shape[0])
np.random.shuffle(indices)
headline_data = headline_data[indices]
article_data = article_data[indices]

stances = stances[indices]
nb_validation_samples = int(VALIDATION_SPLIT * headline_data.shape[0])

headline_train = headline_data[:-nb_validation_samples]
headline_val = headline_data[-nb_validation_samples:]
article_train = article_data[:-nb_validation_samples]
article_val = article_data[-nb_validation_samples:]
y_train = stances[:-nb_validation_samples]
y_val = stances[-nb_validation_samples:]


#####################################
# making the embedding matrices

# headlines
headline_embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.'+str(HEADLINE_EMBEDDING_DIM)+'d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    headline_embeddings_index[word] = coefs
f.close()

print('Found %s headline word vectors.' % len(headline_embeddings_index))

headline_embedding_matrix = np.zeros((len(headline_word_index) + 1, HEADLINE_EMBEDDING_DIM))
for word, i in headline_word_index.items():
    headline_embedding_vector = headline_embeddings_index.get(word)
    if headline_embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        headline_embedding_matrix[i] = headline_embedding_vector

# articles
article_embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.'+str(ARTICLE_EMBEDDING_DIM)+'d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    article_embeddings_index[word] = coefs
f.close()

print('Found %s article word vectors.' % len(article_embeddings_index))

article_embedding_matrix = np.zeros((len(article_word_index) + 1, ARTICLE_EMBEDDING_DIM))
for word, i in article_word_index.items():
    article_embedding_vector = article_embeddings_index.get(word)
    if article_embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        article_embedding_matrix[i] = article_embedding_vector


#########################################
# save outputs to file
outfile = "preProcessed.npz"
np.savez(outfile, \
 headline_train=headline_train, article_train=article_train, y_train=y_train,\
 headline_val=headline_val, article_val=article_val, y_val=y_val, \
 headline_embedding_matrix=headline_embedding_matrix, article_embedding_matrix=article_embedding_matrix,\
 headline_word_index=headline_word_index, article_word_index=article_word_index,\
 headline_embedding_dim=HEADLINE_EMBEDDING_DIM,article_embedding_dim=ARTICLE_EMBEDDING_DIM,\
 max_seq=MAX_SEQUENCE_LENGTH)