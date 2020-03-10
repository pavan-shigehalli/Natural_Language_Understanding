''' This program
- Can be used for training GloVe word embeddings
- Can be used to retrieve GloVe word embeddings for trained words.'''

import os
import pickle
import csv

import tensorflow as tf

from .tf_glove_train import GloVeModel
from .__init__ import Glove as config
#from config import Glove as config


class DataNotfoundError(Exception):
    pass

class NotTrainedError(Exception):
    pass

class GloveEmbeddings():

    def __init__(self, embedding_size=300, # Embedding size of every word
                context_size=10, # Size of the left and the right window
                saved_model=None, # Already saved model if any
                train_file=None, # File to train if any
                max_vocab_size=100000, # Limit on the vocabulary size
                min_occurrences=1, # Minimum occurence of a word within a context window
                scaling_factor=3/4, # Scaling factor
                cooccurrence_cap=100, # Limit on the co-occurence
                batch_size=512, # Batch size
                learning_rate=0.05, # Learning rate
                loss_report_file=None) : # Loss vs iteration file to be saved

        self.tf_saver = saved_model
        self.train_file = train_file
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.max_vocab_size = max_vocab_size
        self.min_occurrences = min_occurrences
        self.scaling_factor = scaling_factor
        self.cooccurrence_cap = cooccurrence_cap
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_report_file = loss_report_file
        self.__corpus = []
        self.__word_to_id = None
        self.__embeddings = None


    @property
    def tf_saver(self) :
        return self._tf_saver

    @tf_saver.setter
    def tf_saver(self,model) :
        if model :
            self._tf_saver = tf.train.import_meta_graph(model + '.meta')
            self.__python_var_save = model + '.pickle'
        else:
            self._tf_saver = None
            self.__python_var_save= None

    @property
    def train_file(self):
        return self._train_file

    @train_file.setter
    def train_file(self,file):
        if file :
            self._train_file = file
        if not file and not self.tf_saver :
            raise DataNotfoundError('Provide either a trained model or data to train the network')


    def tf_train(self, num_epochs, save_model) :
        ''' Trains the model and saves the checkpoints
        in the specified file '''

        self.__form_corpus_from_csv()

        tf_model = GloVeModel(\
        embedding_size=self.embedding_size, \
        context_size=self.context_size, \
        max_vocab_size=self.max_vocab_size,\
        min_occurrences=self.min_occurrences,\
        scaling_factor=self.scaling_factor,\
        cooccurrence_cap=self.cooccurrence_cap,\
        batch_size=self.batch_size,\
        learning_rate=self.learning_rate,\
        loss_report_file=self.loss_report_file)

        tf_model.fit_to_corpus(self.__corpus, save_model + '.pickle')
        tf_model.train(num_epochs=num_epochs, save_model=save_model)


    def __form_corpus_from_txt(self) :
        ''' Forms the corpus from a .txt file. The corpus will be a list of
        list of words
        Ex: [['Lorem', 'ipsum', 'dolor', 'sit', 'amet'] ,['what' ,'does' ,'it' ,'mean?']]
        '''
        file = open(self.train_file).read()
        for sentence in file.split('.') :
            words = []
            for word in sentence.split() :
                # remove all the special characters
                for w in word :
                    if not w.isalnum() :
                        word = word.replace(w,'')
                words.append(word)
            if words :
                self.__corpus.append(words)


    def __form_corpus_from_csv(self) :
        ''' Forms the corpus from a .csv file. The corpus will be a list of
        list of words
        Ex: [['Lorem', 'ipsum', 'dolor', 'sit', 'amet'] ,['what' ,'does' ,'it' ,'mean?']]
        '''
        csv_file = csv.reader(open(self.train_file))
        for i,row in enumerate(csv_file) :
            for j in range(3) :
                words = row[j].split()
                self.__corpus.append(words)


    def load_saved_model(self) :
        ''' Loads the trained model and saved word dictionary '''
        if not self.__python_var_save :
            raise NotTrainedError('Need to train the network before accesing the embeddings')

        with open(self.__python_var_save, 'rb') as f:
            self.__word_to_id = pickle.load(f)

        graph = tf.get_default_graph()
        with tf.Session() as sess:
            self.tf_saver.restore(sess, tf.train.latest_checkpoint(config.MODEL_DIR))
            __combined_embeddings = graph.get_tensor_by_name("combined_embeddings:0")
            self.__embeddings = __combined_embeddings.eval()


    def embedding_for(self, word ):
        ''' Returns the word embeddings '''
        if self.__word_to_id.get(word) :
            return self.__embeddings[self.__word_to_id[word]]
        else:
            return None
