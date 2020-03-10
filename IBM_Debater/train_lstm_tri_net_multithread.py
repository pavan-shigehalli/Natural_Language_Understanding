"""
This program is the implementation of IBM debater train neural network
according to the paper :
Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics
(Short Papers), pages 49–54, Melbourne, Australia,
July 15 - 20, 2018. c 2018 Association for Computational Linguistics

This program parallellizes the data generation/ processing and
network training. The data is pre processed in the background and the
data is always availabale for training.

"""
import itertools
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import math
import random
import csv
import os
import warnings
import pickle
import sys
import time
from datetime import datetime
import logging
import subprocess
from threading import Thread

from .progress import ETA

from .__init__ import Glove as glove_config
from .__init__ import LSTM as lstm_config
from ..GloVe.tf_glove import GloveEmbeddings

logging.basicConfig(filename=lstm_config.TRAIN_LOG, format='[ %(asctime)s ] %(message)s',\
level=logging.INFO)

class NotTrainedError(Exception):
    pass

class miscellaneousError(Exception) :
    pass

class lstm_trip_train():

    def __init__(self, batch_size = 128,  # size of mini batch
                num_of_epochs = 1,  # number of iteration over the dataset
                state_size = 300,  # size of LSTM cell state
                learning_rate = 0.00001,  # Learning rate
                num_of_features = 300,  # size of the vector representing a word
                dropout = 0.8,  # LSTM output dropout
                attention_length = 200, # size of attention nodes
                max_num_of_batches_in_ram = 2,  # maximum number of batches to store in RAM
                train_limit = 8,  # set the cap on number of trainable triplets in the multiple of batch sizes
                max_sentence_length = 50,  # maximum number of words allowed in a sentence
                embedding_size = 300,  # size of the embedding used in GloVe
                num_of_saves_per_epoch = 5) : # number of networks to save per epoch

        self.batch_size = batch_size # number of triplets in one run
        self.num_epochs = num_of_epochs
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.num_of_features = num_of_features
        self.dropout = dropout
        self.attention_length = attention_length

        self.max_sentence_length = max_sentence_length # maximum number of acceptable words in any sentence
        self.embedding_size = embedding_size
        self.num_of_save = num_of_saves_per_epoch

        self.end_of_sen_data_0 = False
        self.end_of_sen_data_1 = False
        self.end_of_title_data_0 = False
        self.end_of_title_data_1 = False
        self.__end_of_data_0 = False
        self.__end_of_data_1 = False
        self.start_train = False

        self.batch_id_list = [0,1]
        self.batch_id = 0
        self.max_batch_in_ram = (max_num_of_batches_in_ram // len(self.batch_id_list))* self.batch_size # maximum number of batches in computer RAM
        self.train_limit =  train_limit # limit in terms of batch size on training data

        #self.__file_trace = {} # stores next data file and the line number in a list
        self.__train_files = {} # stores a list of all data files
        self.__x_batch_0 = []
        self.__xp_batch_0 = []
        self.__xm_batch_0 = []

        self.__x_batch_1 = []
        self.__xp_batch_1 = []
        self.__xm_batch_1 = []

        self.__file_trace_0 = {}
        self.__file_trace_1 = {}
        self.__batch_count_0 = 0
        self.__batch_count_1 = 0
        self.__batch_count = 0 # counts the number of batches or triplets generated

        # load the saved embeddings
        self.__embed = GloveEmbeddings(\
        train_file = None, \
        saved_model = glove_config.SAVED_GLOVE_MODEL )
        self.__embed.load_saved_model()

        # load the saved network
        try:
            with open(lstm_config.SAVED_LSTM + '.pickle') as f :
                self.__tf_saver = True
                self.restore = False
        except FileNotFoundError:
                self.__tf_saver = None
                self.restore = True


    @property
    def train_limit(self) :
        return self._train_limit

    @train_limit.setter
    def train_limit(self,limit) :
        # Set various train limit parameters
        # Read triplet log files
        with open(glove_config.TRIPLET_SEN_LOG) as f :
            next(csv.reader(f))
            triplet_sen_count = int(next(csv.reader(f))[1])
        with open(glove_config.TRIPLET_TITLE_LOG) as f :
            next(csv.reader(f))
            triplet_title_count = int(next(csv.reader(f))[1])
            self.max_train_triplet_count = triplet_sen_count + triplet_title_count

            fact = triplet_sen_count/(self.max_train_triplet_count)
            self.__max_sen_triplet_per_batch = math.ceil(self.batch_size * fact)
            self.__max_title_triplet_per_batch = self.batch_size - self.__max_sen_triplet_per_batch

            if self.__max_title_triplet_per_batch == 0:
                self.__max_title_triplet_per_batch = 1
                self.__max_sen_triplet_per_batch -= 1

            if self.__max_sen_triplet_per_batch == 0:
                self.__max_sen_triplet_per_batch = 1
                self.__max_title_triplet_per_batch -= 1

        if limit is not None :
            self._train_limit = limit * self.batch_size
            self.max_train_triplet_count = limit * self.batch_size
        else:
            true_limit = int(self.batch_size *(self.max_train_triplet_count // self.batch_size))
            self._train_limit = true_limit

        true_limit = int(self.batch_size *(self.max_train_triplet_count // self.batch_size))
        index = math.ceil((true_limit // self.max_batch_in_ram) / self.num_of_save)

        self.save_batches = []
        for i in range(1, self.num_of_save-1) :
            self.save_batches.append(self.max_batch_in_ram * index * i)
        self.save_batches.append(true_limit)

        self.__train_limit_1 = self.batch_size * (self._train_limit // (2*self.batch_size))
        self.__train_limit_0 = self._train_limit - self.__train_limit_1


    def __prepare_files(self):
        ''' Create a list of triplet files to be used for training '''

        path = glove_config.TRIPLET_DIR + glove_config.TRIPLET_SEN_DIR + glove_config.TRIPLET_TRAIN_DIR
        files = sorted(os.listdir(path))
        file_list = []
        for file in files :
            file_list.append(path + file)
        self.__train_files.update({'sentence_file':file_list})

        path  = glove_config.TRIPLET_DIR + glove_config.TRIPLET_TITLE_DIR + glove_config.TRIPLET_TRAIN_DIR
        files = sorted(os.listdir(path))
        file_list = []
        for file in files :
            file_list.append(path + file)
        # train_files stucture {'senetce_file' : [senetnce triplet files], 'title_file' : [ title triplet files ] }
        self.__train_files.update({'title_file':file_list})

        self.__file_trace_0 = {'sentence_file': [self.__train_files['sentence_file'][0], 0], \
        'title_file' :[ self.__train_files['title_file'][0], 0] }

        self.__file_trace_1 = {'sentence_file': [self.__train_files['sentence_file'][0], self.__max_sen_triplet_per_batch], \
        'title_file' :[ self.__train_files['title_file'][0], self.__max_title_triplet_per_batch] }


    def __sentence_to_vector(self, sentence):
        '''' Convert sentence to GloVe word embeddings '''
        time_axis = []
        word_count = 0
        for word in sentence :
            feature_axis = self.__embed.embedding_for(word)
            if feature_axis is not None : # What if the word does not exist ?
                time_axis.append(feature_axis)
                word_count += 1

            if word_count >= self.max_sentence_length :
                warnings.warn('Maximum sentence length is shorter than the number of words in a sentence.\
                network may not be trained properly. Please raise the limit ')
                break # stop reading words in the sentence

        for _ in range(self.max_sentence_length - word_count) : # append zeros
            zeros = [0]*self.embedding_size
            time_axis.append(zeros)

        return time_axis


    def __gen_mini_batch(self, file, start_line, max_count, batch_id) :
        ''' Generate mini batches of size max_count from a file'''

        x_mini_batch = []
        xp_mini_batch = []
        xm_mini_batch = []

        with open(file, encoding='utf-8') as csvfile :
            csv_reader = csv.reader(csvfile)

            for line_count,row in enumerate(csv_reader) :

                if line_count < start_line :
                    continue

                x_mini_batch.append(self.__sentence_to_vector(row[0].split())) # This can be parallelized
                xp_mini_batch.append(self.__sentence_to_vector(row[1].split()))
                xm_mini_batch.append(self.__sentence_to_vector(row[2].split()))
                if batch_id == 0 :
                    self.__batch_count_0 += 1
                if batch_id == 1:
                    self.__batch_count_1 += 1

                if line_count - start_line + 1 == max_count :
                    return x_mini_batch, xp_mini_batch, xm_mini_batch, line_count

        return x_mini_batch, xp_mini_batch, xm_mini_batch, 0 # file was completely read


    def __gen_sentence_batch(self,max_count,batch_id) :
        ''' Generate triplet sentence batch '''
        if batch_id == 0:
            __file_trace = self.__file_trace_0
        elif batch_id == 1 :
            __file_trace = self.__file_trace_1
        else:
            raise miscellaneousError('Batch ID not defined')

        start_line = __file_trace['sentence_file'][1]
        end_of_data = False
        for file_count, file in enumerate(self.__train_files['sentence_file']) :

            if file != __file_trace['sentence_file'][0] :
                continue

            x_mini_batch, xp_mini_batch, xm_mini_batch, start_line = \
            self.__gen_mini_batch(file, start_line, max_count, batch_id)

            if len(x_mini_batch) != max_count :
                end_of_data = True
                __file_trace['sentence_file'] = [self.__train_files['sentence_file'][0], 0]
                return x_mini_batch, xp_mini_batch, xm_mini_batch, end_of_data
            else:
                __file_trace['sentence_file'] = [ file, start_line + \
                (len(self.batch_id_list) - 1) * self.__max_sen_triplet_per_batch]
                return x_mini_batch, xp_mini_batch, xm_mini_batch, end_of_data


    def __gen_title_batch(self,max_count,batch_id) :
        ''' Generate triplet title batch '''

        if batch_id == 0:
            __file_trace = self.__file_trace_0
        elif batch_id == 1 :
            __file_trace = self.__file_trace_1
        else:
            raise miscellaneousError('Batch ID not defined')

        start_line = __file_trace['title_file'][1]
        end_of_data = False
        for file_count, file in enumerate(self.__train_files['title_file']) :

            if file != __file_trace['title_file'][0] :
                continue

            x_mini_batch, xp_mini_batch, xm_mini_batch, start_line = \
            self.__gen_mini_batch(file, start_line, max_count, batch_id)

            if len(x_mini_batch) != max_count :
                end_of_data = True
                __file_trace['title_file'] = [self.__train_files['title_file'][0], 0]
                return x_mini_batch, xp_mini_batch, xm_mini_batch, end_of_data
            else:
                __file_trace['title_file'] = [ file, start_line + \
                (len(self.batch_id_list) - 1) * self.__max_title_triplet_per_batch]
                return x_mini_batch, xp_mini_batch, xm_mini_batch, end_of_data


    def gen_training_batches_0(self,max_batch, batch_id):
        ''' Generates training batches '''

        __x_batch = []
        __xp_batch = []
        __xm_batch = []
        __max_batch_in_ram = False

        while (not __max_batch_in_ram and not self.__end_of_data_0) :

            if not self.end_of_sen_data_0 :
                if not self.end_of_title_data_0 :
                    max_count = self.__max_sen_triplet_per_batch
                else:
                    max_count = self.batch_size
                x_sen_batch, xp_sen_batch, xm_sen_batch, self.end_of_sen_data_0 = self.__gen_sentence_batch(max_count,batch_id)
            else:
                x_sen_batch = []
                xp_sen_batch = []
                xm_sen_batch = []

            if not self.end_of_title_data_0 :
                if not self.end_of_sen_data_0 :
                    max_count = self.__max_title_triplet_per_batch
                else:
                    max_count = self.batch_size
                x_title_batch, xp_title_batch, xm_title_batch, self.end_of_title_data_0 = self.__gen_title_batch(max_count,batch_id)
            else:
                x_title_batch = []
                xp_title_batch = []
                xm_title_batch = []

            if len(x_sen_batch + x_title_batch) == self.batch_size :
                __x_batch.append(x_sen_batch + x_title_batch)
                __xp_batch.append(xp_sen_batch + xp_title_batch)
                __xm_batch.append(xm_sen_batch + xm_title_batch)

            if self.__train_limit_0 is not None and self.__batch_count_0 % self.__train_limit_0 == 0 :
                self.__file_trace_0['sentence_file'] = [self.__train_files['sentence_file'][0], 0]
                self.__file_trace_0['title_file'] = [self.__train_files['title_file'][0], 0]
                self.__end_of_data_0 = True

            if len(__x_batch) * self.batch_size >= max_batch :
                __max_batch_in_ram = True

            if self.end_of_sen_data_0 and self.end_of_title_data_0 :
                self.__end_of_data_0 = True

        return __x_batch, __xp_batch, __xm_batch


    def gen_training_batches_1(self,max_batch, batch_id):
        ''' Generates training batches '''

        __x_batch = []
        __xp_batch = []
        __xm_batch = []
        __max_batch_in_ram = False

        while (not __max_batch_in_ram and not self.__end_of_data_1) :

            if not self.end_of_sen_data_1 :
                if not self.end_of_title_data_1 :
                    max_count = self.__max_sen_triplet_per_batch
                else:
                    max_count = self.batch_size
                x_sen_batch, xp_sen_batch, xm_sen_batch, self.end_of_sen_data_1 = self.__gen_sentence_batch(max_count,batch_id)
            else:
                x_sen_batch = []
                xp_sen_batch = []
                xm_sen_batch = []

            if not self.end_of_title_data_1 :
                if not self.end_of_sen_data_1 :
                    max_count = self.__max_title_triplet_per_batch
                else:
                    max_count = self.batch_size
                x_title_batch, xp_title_batch, xm_title_batch, self.end_of_title_data_1 = self.__gen_title_batch(max_count,batch_id)
            else:
                x_title_batch = []
                xp_title_batch = []
                xm_title_batch = []

            if len(x_sen_batch + x_title_batch) == self.batch_size :
                __x_batch.append(x_sen_batch + x_title_batch)
                __xp_batch.append(xp_sen_batch + xp_title_batch)
                __xm_batch.append(xm_sen_batch + xm_title_batch)

            if self.__train_limit_1 is not None and self.__batch_count_1 % self.__train_limit_1 == 0 :
                self.__file_trace_1['sentence_file'] = [self.__train_files['sentence_file'][0], 0]
                self.__file_trace_1['title_file'] = [self.__train_files['title_file'][0], 0]
                self.__end_of_data_1 = True

            if len(__x_batch) * self.batch_size >= max_batch :
                __max_batch_in_ram = True

            if self.end_of_sen_data_1 and self.end_of_title_data_1 :
                self.__end_of_data_1 = True

        return __x_batch, __xp_batch, __xm_batch


    def gen_data_background(self, cpu_device, batch_id):
        ''' keeps generating data in the background '''
        if cpu_device :
            with tf.device(cpu_device) :
                while True :
                    if batch_id == 0 and not self.__end_of_data_0:
                        max_batch = self.max_batch_in_ram - len(self.__x_batch_0) * self.batch_size
                        x,xp,xm = self.gen_training_batches_0(max_batch, 0)

                        self.__x_batch_0 += x
                        self.__xp_batch_0 += xp
                        self.__xm_batch_0 += xm
                        self.start_train = True
                        current_batch_size = len(self.__x_batch_0)
                        while len(self.__x_batch_0) == current_batch_size :
                            # wait till the list size shrinks
                            time.sleep(0.1)

                    if batch_id == 1 and not self.__end_of_data_1:
                        max_batch = self.max_batch_in_ram - len(self.__x_batch_1) * self.batch_size
                        x,xp,xm = self.gen_training_batches_1(max_batch, 1)

                        self.__x_batch_1 += x
                        self.__xp_batch_1 += xp
                        self.__xm_batch_1 += xm
                        self.start_train = True
                        current_batch_size = len(self.__x_batch_1)
                        while len(self.__x_batch_1) == current_batch_size :
                            # wait till the list size shrinks
                            time.sleep(0.1)
                    time.sleep(0.001)
        else:
            while True:
                if batch_id == 0 and not self.__end_of_data_0:
                    max_batch = self.max_batch_in_ram - len(self.__x_batch_0) * self.batch_size
                    x,xp,xm = self.gen_training_batches_0(max_batch, 0)

                    self.__x_batch_0 += x
                    self.__xp_batch_0 += xp
                    self.__xm_batch_0 += xm
                    self.start_train = True
                    current_batch_size = len(self.__x_batch_0)
                    while len(self.__x_batch_0) == current_batch_size :
                        # wait till the list size shrinks
                        time.sleep(0.1)

                if batch_id == 1 and not self.__end_of_data_1:
                    max_batch = self.max_batch_in_ram - len(self.__x_batch_1) * self.batch_size
                    x,xp,xm = self.gen_training_batches_1(max_batch, 1)

                    self.__x_batch_1 += x
                    self.__xp_batch_1 += xp
                    self.__xm_batch_1 += xm
                    self.start_train = True
                    current_batch_size = len(self.__x_batch_1)
                    while len(self.__x_batch_1) == current_batch_size :
                        # wait till the list size shrinks
                        time.sleep(0.1)
                time.sleep(0.001)


    def __lstm_state_tuple(self,state_fw_placeholder, state_bw_placeholder) :
        ''' Return LSTM tuple containing cell state and hidden state for forward
        and backward cells '''
        s1 = tf.unstack(state_fw_placeholder, axis=0)
        s2 = tf.unstack(state_bw_placeholder, axis=0)
        state_fw = tf.nn.rnn_cell.LSTMStateTuple(s1[0],s1[1])
        state_bw = tf.nn.rnn_cell.LSTMStateTuple(s2[0],s2[1])

        return state_fw, state_bw


    def attention_layer(self, hidden_state, name='attention') :
        ''' Attention layer '''
        with tf.name_scope(name) :
            alpha = []
            for i in range(self.batch_size) :
                U = tf.math.tanh(tf.math.add(tf.linalg.matmul(self.W, hidden_state[i], transpose_b=True),self.b))
                logits = tf.linalg.matmul(U, self.Uw, transpose_a=True)
                alpha.append(tf.nn.softmax(logits))
            time_attention = tf.linalg.matmul(alpha, hidden_state, transpose_a=True)

            return time_attention


    def build_lstm(self):
        ''' Build the training graph '''

        self.batchX_placeholder = tf.placeholder(tf.float64, [self.batch_size,self.max_sentence_length,self.num_of_features], name='X_batch')
        self.batchXp_placeholder = tf.placeholder(tf.float64, [self.batch_size,self.max_sentence_length,self.num_of_features], name='Xp_batch')
        self.batchXm_placeholder = tf.placeholder(tf.float64, [self.batch_size,self.max_sentence_length,self.num_of_features], name='Xm_batch')
        self.statex_fw_placeholder = tf.placeholder(tf.float64,[2,self.batch_size, self.state_size], name='state_x_fw')
        self.statex_bw_placeholder = tf.placeholder(tf.float64,[2,self.batch_size, self.state_size], name='state_x_bw')
        self.statexp_fw_placeholder = tf.placeholder(tf.float64,[2,self.batch_size, self.state_size], name='state_xp_fw')
        self.statexp_bw_placeholder = tf.placeholder(tf.float64,[2,self.batch_size, self.state_size], name='state_xp_bw')
        self.statexm_fw_placeholder = tf.placeholder(tf.float64,[2,self.batch_size, self.state_size],name='state_xm_fw')
        self.statexm_bw_placeholder = tf.placeholder(tf.float64,[2,self.batch_size, self.state_size], name='state_xm_bw')
        self.W_placeholder = tf.placeholder(tf.float64, [self.attention_length,2*self.state_size], name='W')
        self.b_placeholder = tf.placeholder(tf.float64, [self.attention_length,self.max_sentence_length], name='b')
        self.Uw_placeholder = tf.placeholder(tf.float64, [self.attention_length,1],name='Uw')

        tf.assign(self.W,self.W_placeholder)
        tf.assign(self.b,self.b_placeholder)
        tf.assign(self.Uw, self.Uw_placeholder)

        # State X LSTM cells
        with tf.name_scope('X_net') :
            self.cellx_fw = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True, name='lstm_fw', dtype=tf.float64)
            self.cellx_fwd = tf.nn.rnn_cell.DropoutWrapper(self.cellx_fw, output_keep_prob=self.dropout)
            self.cellx_bw = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True, name='lstm_bw', dtype=tf.float64)
            self.cellx_bwd = tf.nn.rnn_cell.DropoutWrapper(self.cellx_bw, output_keep_prob=self.dropout)
            state_fw, state_bw = self.__lstm_state_tuple(self.statex_fw_placeholder, self.statex_bw_placeholder)

            self.outputx, (self.statex_fw, self.statex_bw) = tf.nn.bidirectional_dynamic_rnn(self.cellx_fwd, self.cellx_bwd, \
            self.batchX_placeholder, initial_state_fw=state_fw, initial_state_bw=state_bw, dtype=tf.float64)
            hx = tf.concat(self.outputx, axis=2)
            outx = self.attention_layer(hx)

        # State X+ LSTM cells
        with tf.name_scope('Xp_net') :
            cellxp_fw = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True, reuse=True, name='lstm_fw', dtype=tf.float64)
            cellxp_fw = tf.nn.rnn_cell.DropoutWrapper(cellxp_fw, output_keep_prob=self.dropout)
            cellxp_bw = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True, reuse=True, name='lstm_bw', dtype=tf.float64)
            cellxp_bw = tf.nn.rnn_cell.DropoutWrapper(cellxp_bw, output_keep_prob=self.dropout)
            state_fw, state_bw = self.__lstm_state_tuple(self.statexp_fw_placeholder, self.statexp_bw_placeholder)
            outputxp,(self.statexp_fw, self.statexp_bw) = tf.nn.bidirectional_dynamic_rnn(cellxp_fw, cellxp_bw, \
            self.batchXp_placeholder, initial_state_fw=state_fw,initial_state_bw=state_bw, dtype=tf.float64)
            hxp = tf.concat(outputxp, axis=2) # dimension = [batch size ,50 ,600]
            outxp = self.attention_layer(hxp) # dimension = [batch size ,1 ,600]

        # State X- LSTM cells
        with tf.name_scope('Xm_net') :
            cellxm_fw = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True, reuse=True, name='lstm_fw', dtype=tf.float64)
            cellxm_fw = tf.nn.rnn_cell.DropoutWrapper(cellxm_fw, output_keep_prob=self.dropout)
            cellxm_bw = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True, reuse=True, name='lstm_bw', dtype=tf.float64)
            cellxm_bw = tf.nn.rnn_cell.DropoutWrapper(cellxm_bw, output_keep_prob=self.dropout)
            state_fw, state_bw = self.__lstm_state_tuple(self.statexm_fw_placeholder, self.statexm_bw_placeholder)
            outputxm, (self.statexm_fw, self.statexm_bw) = tf.nn.bidirectional_dynamic_rnn(cellxm_fw, cellxm_bw, \
            self.batchXm_placeholder, initial_state_fw=state_fw,initial_state_bw=state_bw, dtype=tf.float64)
            hxm = tf.concat(outputxm, axis=2)
            self.outxm = self.attention_layer(hxm)

        with tf.name_scope('dist_p') :
            distp = tf.reduce_sum(tf.abs(tf.subtract(outx, outxp)),axis=2) # dimension = [batch size,1]
        with tf.name_scope('dist_m') :
            distm = tf.reduce_sum(tf.abs(tf.subtract(outx, self.outxm)),axis=2) # dimension = [batch size,1]

        with tf.name_scope('Loss') :
            prob = tf.nn.softmax(tf.concat([distp, distm],axis=1),axis=1) # dimension = [ batch size, 2]
            probp, probm = tf.split(prob, [1,1], axis=1) # dimension = [ batch size, 1], [ batch size, 1]
            ones = tf.constant(1.0, shape=(self.batch_size,1),dtype = tf.float64)
            self.loss = tf.abs(probp) + tf.abs(tf.cast(ones,dtype=tf.float64) - probm) # Its just  0 =< 2*prob[0] <= 2
        with tf.name_scope('accuracy') :
            const = tf.constant(2.0, shape=(1,1),dtype=tf.float64)
            avg_loss = tf.math.divide(tf.reduce_sum(self.loss),self.batch_size)
            self.accuracy = tf.math.multiply(tf.math.divide(const - avg_loss, const), 100 )
            tf.summary.scalar('accuracy',self.accuracy)
        with tf.name_scope('Train') :
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def train_lstm(self):
        ''' Train the graph '''
        self.W = tf.get_variable('weight', initializer = np.zeros((self.attention_length,2*self.state_size)), dtype=tf.float64)
        self.b = tf.get_variable('bias', initializer = np.zeros((self.attention_length,self.max_sentence_length)), dtype=tf.float64)
        self.Uw = tf.get_variable('U_weight', initializer = np.zeros((self.attention_length,1)), dtype=tf.float64)

        self.build_lstm()

        # Initialise the LSTM states
        if self.__tf_saver is not None :
            with open(lstm_config.SAVED_LSTM + '.pickle', 'rb') as f:
                [self.__train_files,  self.__file_trace_0,self.__file_trace_1, batch_count_0,batch_count_1,last_epoch,external_weight, weight_attn,\
                bias_attn, weightf, weightb, _statex_fw,_statex_bw , _statexp_fw,_statexp_bw,_statexm_fw,_statexm_bw,_,_] = pickle.load(f)

            self.__batch_count_0 = batch_count_0
            self.__batch_count_1 = batch_count_1
            if self.__batch_count_0 == self.__train_limit_0 and self.__batch_count_1 == self.__train_limit_1 : # last epoch was complete
                self.__batch_count_0 = 0
                self.__batch_count_1 = 0
                last_epoch += 1

            logging.info('RESUMING THE TRAINING ')

        else:
            self.__prepare_files()
            zero_state = np.zeros((2,self.batch_size, self.state_size))
            _statex_fw = zero_state
            _statex_bw = zero_state
            _statexp_fw = zero_state
            _statexp_bw = zero_state
            _statexm_fw = zero_state
            _statexm_bw = zero_state
            weight_attn = np.zeros((self.attention_length,2*self.state_size))
            bias_attn =  np.zeros((self.attention_length,self.max_sentence_length))
            external_weight =  np.zeros((self.attention_length,1))
            last_epoch = 0
            logging.info('NO PRE-TRAINED DATA FOUND. STARTING FRESH TRAINING ')

        logging.info('Starting training with the files : 0 :{} , {}, 1: {}, {}'\
        .format(self.__file_trace_0['sentence_file'], self.__file_trace_0['title_file'],\
        self.__file_trace_1['sentence_file'], self.__file_trace_1['title_file']))

        logging.info('Save batches : {}'.format(self.save_batches))
        logging.info('Train limits : {} , {}'.format(self.__train_limit_0,self.__train_limit_1))
        logging.info('Max sentence triplet per batch : {} , Max title triplet per batch : {} '\
        .format(self.__max_sen_triplet_per_batch,self.__max_title_triplet_per_batch))

        devices = device_lib.list_local_devices()
        logging.info('detected devices : {}'.format(devices))
        cpu_devices = []
        for dev in devices :
            if dev.device_type == "CPU" or dev.device_type == "XLA_CPU" :
                cpu_devices.append(dev.name)
        logging.info('Generating batches with CPU : {}'.format(cpu_devices))

        bar = ETA(self.num_epochs * self.max_train_triplet_count)
        bar.start()

        with tf.Session() as sess: # config=tf.ConfigProto(log_device_placement=True)
            tf.global_variables_initializer().run()
            merged_summary = tf.summary.merge_all()
            #writer = tf.summary.FileWriter("/tmp/TensorBoard", sess.graph)
            Thread(target=self.gen_data_background, args=(cpu_devices[0 % len(cpu_devices)],0), daemon=True).start()
            Thread(target=self.gen_data_background, args=(cpu_devices[1 % len(cpu_devices)],1), daemon=True).start()

            for i in range(last_epoch , self.num_epochs) :

                self.end_of_sen_data_0 = False
                self.end_of_sen_data_1 = False
                self.end_of_title_data_0 = False
                self.end_of_title_data_1 = False
                self.__end_of_data_0 = False
                self.__end_of_data_1 = False
                self.start_train = False
                batch_id_count = 0
                while True :

                    if not self.restore :
                        self.restore = True
                        for in_x in np.zeros((1,self.batch_size,self.max_sentence_length,self.embedding_size)) :
                            sess.run(self.outputx,\
                            feed_dict={self.batchX_placeholder : in_x,\
                            self.statex_fw_placeholder : _statex_fw,\
                            self.statex_bw_placeholder : _statex_bw })

                        self.cellx_fw.set_weights(weightf)
                        self.cellx_bw.set_weights(weightb)

                    t1 = time.time()
                    while not self.start_train :
                        time.sleep(0.1)

                    logging.info('Initialize time : {}'.format(round(time.time()-t1,3)))
                    t1 = time.time()
                    if batch_id_count == 0:
                        __x_batch = self.__x_batch_0
                        __xp_batch = self.__xp_batch_0
                        __xm_batch = self.__xm_batch_0

                        while len(np.array(__x_batch).shape) != 4  :
                            if self.__end_of_data_0 :
                                break
                            time.sleep(0.1)

                    if batch_id_count == 1:
                        __x_batch = self.__x_batch_1
                        __xp_batch = self.__xp_batch_1
                        __xm_batch = self.__xm_batch_1

                        while len(np.array(__x_batch).shape) != 4  :
                            if self.__end_of_data_1 :
                                break
                            time.sleep(0.1)

                    if self.__end_of_data_0 and self.__end_of_data_1 :
                        time.sleep(0.5)
                        if len(np.array(__x_batch).shape) != 4 :
                            break

                    train_counter = 0
                    logging.info('Training data from thread : {} with data size : {}'.format(batch_id_count,np.array(__x_batch).shape))
                    for in_x, in_xp, in_xm in zip(__x_batch,__xp_batch,__xm_batch) :
                        external_weight,weight_attn,bias_attn,loss,_, _statex_fw, _statex_bw, _statexp_fw, _statexp_bw, \
                        _statexm_fw, _statexm_bw,accuracy = \
                        sess.run([self.Uw,self.W,self.b,self.loss, self.optimizer, self.statex_fw, self.statex_bw,\
                        self.statexp_fw, self.statexp_bw, self.statexm_fw, self.statexm_bw,self.accuracy],\
                        feed_dict={self.batchX_placeholder : in_x,\
                        self.batchXp_placeholder : in_xp,\
                        self.batchXm_placeholder : in_xm,\
                        self.statex_fw_placeholder : _statex_fw,\
                        self.statex_bw_placeholder : _statex_bw,\
                        self.statexp_fw_placeholder : _statexp_fw, \
                        self.statexp_bw_placeholder : _statexp_bw,\
                        self.statexm_fw_placeholder : _statexm_fw, \
                        self.statexm_bw_placeholder : _statexm_bw,\
                        self.W_placeholder : weight_attn,\
                        self.b_placeholder : bias_attn,\
                        self.Uw_placeholder : external_weight})

                        train_counter += 1

                    del __x_batch[0:train_counter]
                    del __xp_batch[0:train_counter]
                    del __xm_batch[0:train_counter]

                    batch_id_count = (batch_id_count + 1) % len(self.batch_id_list)

                    logging.info('Training time : {}'.format(round(time.time()-t1,3)))
                    eta = bar.update(self.max_train_triplet_count * i + self.__batch_count_0 + self.__batch_count_1)
                    logging.info('{} % progress    ---- ETA : {} '.format(eta[0],eta[1]))

                    weightf = self.cellx_fw.get_weights()
                    weightb = self.cellx_bw.get_weights()
                    t1 = time.time()
                    with open(lstm_config.NEW_LSTM + '.pickle','wb') as f :
                        pickle.dump([self.__train_files, self.__file_trace_0,self.__file_trace_1, self.__batch_count_0,self.__batch_count_1,i, external_weight, weight_attn, bias_attn,\
                        weightf, weightb, _statex_fw,_statex_bw ,_statexp_fw,_statexp_bw,_statexm_fw,_statexm_bw,loss,accuracy], f)
                    logging.info('Data saving cost : {}'.format(round(time.time()-t1,3)))

                    logging.info('Saving network with batch count_0 = {}, batch count_1 = {}, Total = {}'.\
                    format(self.__batch_count_0,self.__batch_count_1,self.__batch_count_0 + self.__batch_count_1))

                    if self.__batch_count in self.save_batches :
                        os.system('mkdir -p ' + lstm_config.NEW_LSTM_FILE_STACK + 'epoch_' + str(i))
                        with open(lstm_config.NEW_LSTM_FILE_STACK + 'epoch_' + str(i) + '/model_' + str(self.__batch_count) + '.pickle','wb') as f :
                            pickle.dump([self.__train_files,  self.__file_trace_0,self.__file_trace_1, self.__batch_count_0,self.__batch_count_1,i, external_weight, weight_attn, bias_attn,\
                            weightf, weightb, _statex_fw,_statex_bw ,_statexp_fw,_statexp_bw,_statexm_fw,_statexm_bw,loss,accuracy], f)

                        logging.info('Saving network in datastack with batch count_0 = {}, batch count_1 = {}, Total = {}'.\
                        format(self.__batch_count_0,self.__batch_count_1,self.__batch_count_0 + self.__batch_count_1))

                    # Save the network
                    tf.train.Saver().save(sess, lstm_config.NEW_LSTM)

                self.__batch_count = 0 # reset the batch count
                self.__batch_count_0 = 0
                self.__batch_count_1 = 0
                #tf.add_summary(summary,i)

            #writer.close()
            eta = bar.finish()
            logging.info('{} : {} % progress    ---- ETA : {} '.format(datetime.now().strftime("%Y-%m-%d;%H:%M:%S"),eta[0],eta[1]))
            print('Oh. Finally, its done !')
            pid = subprocess.check_output('ps ax | grep train.sh' ,shell=True).split()[0].decode('utf-8')
            os.system('kill ' + pid)


if __name__ == '__main__' :
    train = lstm_trip_train()
    train.train_lstm()
