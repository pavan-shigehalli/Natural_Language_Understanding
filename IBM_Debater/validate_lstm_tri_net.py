import itertools
import tensorflow as tf
import numpy as np
import random
import csv
import os
import warnings
import pickle
import sys
import time
import math
import logging

import progressbar
import psutil

from .__init__ import LSTM as lstm_config
from ..GloVe.tf_glove import GloveEmbeddings
from ..GloVe.__init__ import Glove as glove_config

logging.basicConfig(filename=lstm_config.REPORT_LOG, level=logging.INFO, format='%(levelname)s: %(asctime)s : %(message)s')

class Validate_LSTM():
    ''' Validates the trained network '''

    def __init__(self, validation_limit=1024, # Limit on the validation. Should be multiple of max_triplet_in_ram
                max_triplet_in_ram = 1024, # Maximum number of triplets in RAM
                state_size=300, # LSTM cell state size
                max_sentence_length=50, # Maximum number of words in a sentence
                dropout = 0.8, # Dropout factor
                num_of_features=300,
                learning_rate=0.1, # Learning rate
                embedding_size=300, # embedding size
                attention_length = 200, # size of the attention layer
                num_of_epochs = 300): # Number of epochs

        self.validation_limit = validation_limit # maximum number of triplets to verify
        self.max_triplet_in_ram = max_triplet_in_ram  # maximum number of triplets in computer RAM

        self.state_size = state_size
        self.batch_size = 1
        self.max_sentence_length = max_sentence_length
        self.dropout = dropout
        self.num_of_features = num_of_features
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.attention_length = attention_length
        self.num_of_epochs = num_of_epochs

        self.pass_count = 0
        self.fail_count = 0

        self.__file_trace = {}
        self.__validation_files = {}
        self.__triplet_count = 0
        self.__batch_count = 0
        self.__end_of_data = False

        # load embedding model
        self.__embed = GloveEmbeddings(\
        train_file = None, \
        saved_model = glove_config.SAVED_GLOVE_MODEL )
        self.__embed.load_saved_model()

        with open(glove_config.TRIPLET_SEN_LOG) as f :
            next(csv.reader(f))
            next(csv.reader(f))
            triplet_sen_count = int(next(csv.reader(f))[1])
        with open(glove_config.TRIPLET_TITLE_LOG) as f :
            next(csv.reader(f))
            next(csv.reader(f))
            triplet_title_count = int(next(csv.reader(f))[1])
            self.max_validate_triplet_count = triplet_sen_count + triplet_title_count

            fact = triplet_sen_count/(self.max_validate_triplet_count)
            self.__max_sen_triplet_in_ram = math.ceil(self.max_triplet_in_ram * fact)
            self.__max_title_triplet_in_ram = self.max_triplet_in_ram - self.__max_sen_triplet_in_ram

            if self.validation_limit is not None :
                self.max_validate_triplet_count = self.validation_limit


        # load saved network
        '''try:
            self.__tf_saver = tf.train.import_meta_graph(lstm_config.SAVED_LSTM + '.meta')
        except OSError : # saved file does not exists
            raise NotTrainedError('NO PRE-TRAINED NETWORK FOUND')'''


    @property
    def validation_limit(self):
        return self._validation_limit

    @validation_limit.setter
    def validation_limit(self, limit):
        if limit is not None :
            self._validation_limit = limit
        else:
            self._validation_limit = None


    def __prepare_files(self):
        ''' prepare the files for validation '''

        path = glove_config.TRIPLET_DIR + glove_config.TRIPLET_SEN_DIR + glove_config.TRIPLET_VALID_DIR
        files = os.listdir(path)
        file_list = []
        for file in files :
            file_list.append(path + file)
        self.__validation_files.update({'sentence_file':file_list})

        path  = glove_config.TRIPLET_DIR + glove_config.TRIPLET_TITLE_DIR + glove_config.TRIPLET_VALID_DIR
        files = os.listdir(path)
        file_list = []
        for file in files :
            file_list.append(path + file)
        self.__validation_files.update({'title_file':file_list})

        self.__file_trace = {'sentence_file': [self.__validation_files['sentence_file'][0], 0], \
        'title_file' :[ self.__validation_files['title_file'][0], 0] }


    def __gen_single_batch(self, sentence):
        ''' generate single batch '''
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


    def __gen_validation_triplets(self):
        ''' Generates training batches '''

        self.__x_list = []
        self.__xp_list = []
        self.__xm_list = []

        __validation_limit = False
        __max_triplet_in_ram = False

        for file_count in range(len(self.__validation_files)) :

            if self.__validation_files[file_count] != self.__file_trace[0] :
                continue

            csvfile = open(self.__file_trace[0])
            csv_reader = csv.reader(csvfile)

            for line_count,row in enumerate(csv_reader) :

                if __max_triplet_in_ram or __validation_limit :
                    break
                if line_count < self.__file_trace[1] : # read after the last line
                    continue

                self.__x_list.append(self.__gen_single_batch(row[0].split())) # This can be parallelized
                self.__xp_list.append(self.__gen_single_batch(row[1].split()))
                self.__xm_list.append(self.__gen_single_batch(row[2].split()))
                self.__triplet_count += 1


                if len(self.__x_list) * self.batch_size >= self.max_triplet_in_ram :
                    self.__file_trace[1] = line_count + 1 # next line for next set of batches
                    __max_triplet_in_ram = True

                if self.validation_limit is not None and self.__triplet_count % self.validation_limit == 0 :
                    self.__file_trace[1] = line_count + 1 # next line for next set of batches
                    __validation_limit = True

            if not __max_triplet_in_ram and not __validation_limit : # the previous file was completely read
                if file_count + 1 < len(self.__validation_files) : # if next file exists
                    self.__file_trace[0] = self.__validation_files[file_count + 1]
                    self.__file_trace[1] = 0
                else: # End of training data
                    self.__end_of_data = True
                    self.__file_trace[0] = self.__validation_files[0]
                    self.__file_trace[1] = 0

            if __validation_limit :
                self.__end_of_data = True

            if __max_triplet_in_ram or __validation_limit :
                break


    def __sentence_to_vector(self, sentence):
        '''' Generates the single batch of data '''
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


    def __gen_mini_batch(self, file, start_line, max_count) :

        x_mini_batch = []
        xp_mini_batch = []
        xm_mini_batch = []

        csvfile = open(file)
        csv_reader = csv.reader(csvfile)

        for line_count,row in enumerate(csv_reader) :

            if line_count < start_line :
                continue

            x_mini_batch.append(self.__sentence_to_vector(row[0].split())) # This can be parallelized
            xp_mini_batch.append(self.__sentence_to_vector(row[1].split()))
            xm_mini_batch.append(self.__sentence_to_vector(row[2].split()))
            self.__batch_count += 1

            if line_count - start_line + 1 == max_count :
                return x_mini_batch, xp_mini_batch, xm_mini_batch, line_count

        return x_mini_batch, xp_mini_batch, xm_mini_batch, 0 # file was completely read


    def __gen_sentence_batch(self) :
        # generate triplet senetnce batch
        max_count = self.__max_sen_triplet_in_ram
        start_line = self.__file_trace['sentence_file'][1]
        end_of_data = False
        for file_count, file in enumerate(self.__validation_files['sentence_file']) :

            if file != self.__file_trace['sentence_file'][0] :
                continue

            x_mini_batch, xp_mini_batch, xm_mini_batch, start_line = \
            self.__gen_mini_batch(file, start_line, max_count)

            if len(x_mini_batch) != max_count :
                if file_count + 1 < len(self.__validation_files['sentence_file']) : # next file exists
                    self.__file_trace['sentence_file'] = [self.__validation_files['sentence_file'][file_count + 1], start_line]
                else:
                    end_of_data = True
                    self.__file_trace['sentence_file'] = [self.__validation_files['sentence_file'][0], 0]
                    return x_mini_batch, xp_mini_batch, xm_mini_batch, end_of_data
            else:
                self.__file_trace['sentence_file'] = [ file, start_line]
                return x_mini_batch, xp_mini_batch, xm_mini_batch, end_of_data


    def __gen_title_batch(self) :
        # generate triplet senetnce batch
        max_count = self.__max_title_triplet_in_ram
        start_line = self.__file_trace['title_file'][1]
        end_of_data = False
        for file_count, file in enumerate(self.__validation_files['title_file']) :

            if file != self.__file_trace['title_file'][0] :
                continue

            x_mini_batch, xp_mini_batch, xm_mini_batch, start_line = \
            self.__gen_mini_batch(file, start_line, max_count)

            if len(x_mini_batch) != max_count :
                if file_count + 1 < len(self.__validation_files['title_file']) :
                    self.__file_trace['title_file'] = [self.__validation_files['title_file'][file_count + 1], start_line]
                else :
                    end_of_data = True
                    self.__file_trace['title_file'] = [self.__validation_files['title_file'][0], 0]
                    return x_mini_batch, xp_mini_batch, xm_mini_batch, end_of_data
            else:
                self.__file_trace['title_file'] = [ file, start_line]
                return x_mini_batch, xp_mini_batch, xm_mini_batch, end_of_data


    def __gen_validation_batches(self):
        ''' Generates training batches '''

        self.__x_batch = []
        self.__xp_batch = []
        self.__xm_batch = []

        __max_triplet_in_ram = False
        end_of_sen_data = False
        end_of_title_data = False

        while (not __max_triplet_in_ram and not self.__end_of_data) :

            if not end_of_sen_data :
                x_sen_batch, xp_sen_batch, xm_sen_batch, end_of_sen_data = self.__gen_sentence_batch()
            if not end_of_title_data :
                x_title_batch, xp_title_batch, xm_title_batch, end_of_title_data = self.__gen_title_batch()

            self.__x_batch = x_sen_batch + x_title_batch
            self.__xp_batch = xp_sen_batch + xp_title_batch
            self.__xm_batch = xm_sen_batch + xm_title_batch


            if self.validation_limit is not None and self.__batch_count % self.validation_limit == 0 :
                self.__file_trace['sentence_file'] = [self.__validation_files['sentence_file'][0], 0]
                self.__file_trace['title_file'] = [self.__validation_files['title_file'][0], 0]
                self.__end_of_data = True

            if len(self.__x_batch) >= self.max_triplet_in_ram :
                __max_triplet_in_ram = True

            if end_of_sen_data and end_of_title_data :
                self.__end_of_data = True

        return


    def __lstm_state_tuple(self,state_fw_placeholder, state_bw_placeholder) :
        s1 = tf.unstack(state_fw_placeholder, axis=0)
        s2 = tf.unstack(state_bw_placeholder, axis=0)
        state_fw = tf.nn.rnn_cell.LSTMStateTuple(s1[0],s1[1])
        state_bw = tf.nn.rnn_cell.LSTMStateTuple(s2[0],s2[1])

        return state_fw, state_bw


    def attention_layer(self, hidden_state) :
        # Attention layer
        alpha = []
        for i in range(self.batch_size) :
            U = tf.math.tanh(tf.math.add(tf.linalg.matmul(self.W, hidden_state[i], transpose_b=True),self.b))
            logits = tf.linalg.matmul(U, self.Uw, transpose_a=True)
            alpha.append(tf.nn.softmax(logits))
        time_attention = tf.linalg.matmul(alpha, hidden_state, transpose_a=True)

        return time_attention


    def __build_lstm(self):
        self.batchX_placeholder = tf.placeholder(tf.float64, [self.batch_size,self.max_sentence_length,self.num_of_features])
        self.batchXp_placeholder = tf.placeholder(tf.float64, [self.batch_size,self.max_sentence_length,self.num_of_features])
        self.batchXm_placeholder = tf.placeholder(tf.float64, [self.batch_size,self.max_sentence_length,self.num_of_features])
        self.statex_fw_placeholder = tf.placeholder(dtype=tf.float64,shape=(2,self.batch_size, self.state_size))
        self.statex_bw_placeholder = tf.placeholder(dtype=tf.float64,shape=(2,self.batch_size, self.state_size))
        self.statexp_fw_placeholder = tf.placeholder(dtype=tf.float64,shape=(2,self.batch_size, self.state_size))
        self.statexp_bw_placeholder = tf.placeholder(dtype=tf.float64,shape=(2,self.batch_size, self.state_size))
        self.statexm_fw_placeholder = tf.placeholder(dtype=tf.float64,shape=(2,self.batch_size, self.state_size))
        self.statexm_bw_placeholder = tf.placeholder(dtype=tf.float64,shape=(2,self.batch_size, self.state_size))


        # State X LSTM cells
        self.cellx_fw = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True, name='lstm_fw', dtype=tf.float64)
        self.cellx_fwd = tf.nn.rnn_cell.DropoutWrapper(self.cellx_fw, output_keep_prob=self.dropout)

        self.cellx_bw = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True, name='lstm_bw', dtype=tf.float64)
        self.cellx_bwd = tf.nn.rnn_cell.DropoutWrapper(self.cellx_bw, output_keep_prob=self.dropout)

        state_fw, state_bw = self.__lstm_state_tuple(self.statex_fw_placeholder, self.statex_bw_placeholder)
        outputx, (self.statex_fw, self.statex_bw) = tf.nn.bidirectional_dynamic_rnn(self.cellx_fwd, self.cellx_bwd, \
        self.batchX_placeholder, initial_state_fw=state_fw, initial_state_bw=state_bw, dtype=tf.float64)
        outx = tf.concat(outputx, axis=-1)

        # State X+ LSTM cells
        cellxp_fw = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True, reuse=True, name='lstm_fw', dtype=tf.float64)
        cellxp_fw = tf.nn.rnn_cell.DropoutWrapper(cellxp_fw, output_keep_prob=self.dropout)

        cellxp_bw = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True, reuse=True, name='lstm_bw', dtype=tf.float64)
        cellxp_bw = tf.nn.rnn_cell.DropoutWrapper(cellxp_bw, output_keep_prob=self.dropout)

        state_fw, state_bw = self.__lstm_state_tuple(self.statexp_fw_placeholder, self.statexp_bw_placeholder)
        outputxp,(self.statexp_fw, self.statexp_bw) = tf.nn.bidirectional_dynamic_rnn(cellxp_fw, cellxp_bw, \
        self.batchXp_placeholder, initial_state_fw=state_fw,initial_state_bw=state_bw, dtype=tf.float64)
        outxp = tf.concat(outputxp, axis=-1)

        # State X- LSTM cells
        cellxm_fw = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True, reuse=True, name='lstm_fw', dtype=tf.float64)
        cellxm_fw = tf.nn.rnn_cell.DropoutWrapper(cellxm_fw, output_keep_prob=self.dropout)

        cellxm_bw = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True, reuse=True, name='lstm_bw', dtype=tf.float64)
        cellxm_bw = tf.nn.rnn_cell.DropoutWrapper(cellxm_bw, output_keep_prob=self.dropout)

        state_fw, state_bw = self.__lstm_state_tuple(self.statexm_fw_placeholder, self.statexm_bw_placeholder)
        outputxm, (self.statexm_fw, self.statexm_bw) = tf.nn.bidirectional_dynamic_rnn(cellxm_fw, cellxm_bw, \
        self.batchXm_placeholder, initial_state_fw=state_fw,initial_state_bw=state_bw, dtype=tf.float64)
        outxm = tf.concat(outputxm, axis=-1)

        distp = tf.reduce_sum(tf.reduce_sum(tf.abs(tf.subtract(outx, outxp)),axis=1),axis=1)
        distm = tf.reduce_sum(tf.reduce_sum(tf.abs(tf.subtract(outx, outxm)),axis=1),axis=1)

        prob = tf.nn.softmax(tf.to_float([distp, distm]),axis=0)
        ones = tf.constant(1.0, shape=(1,self.batch_size),dtype = tf.float64)
        self.loss = tf.abs(prob[0]) + tf.abs(tf.to_float(ones) - prob[1]) # Its just  0 =< 2*prob[0] <= 2
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def __build_test(self):

        zero_state = np.zeros((2,self.batch_size, self.state_size))
        _state_fw, _state_bw = self.__lstm_state_tuple(zero_state, zero_state)
        self.X_placeholder = tf.placeholder(tf.float64, [self.batch_size,self.max_sentence_length,self.num_of_features])
        self.Xp_placeholder = tf.placeholder(tf.float64, [self.batch_size,self.max_sentence_length,self.num_of_features])
        self.Xm_placeholder = tf.placeholder(tf.float64, [self.batch_size,self.max_sentence_length,self.num_of_features])

        outputx, _ = tf.nn.bidirectional_dynamic_rnn(self.cellx_fw, self.cellx_bw, \
        self.X_placeholder, initial_state_fw=_state_fw, initial_state_bw=_state_bw, dtype=tf.float64)
        hx = tf.concat(outputx, axis=2)
        outx = self.attention_layer(hx)

        outputxp, _ = tf.nn.bidirectional_dynamic_rnn(self.cellx_fw, self.cellx_bw, \
        self.Xp_placeholder, initial_state_fw=_state_fw, initial_state_bw=_state_bw, dtype=tf.float64)
        hxp = tf.concat(outputxp, axis=2) # dimension = [batch size ,50 ,600]
        outxp = self.attention_layer(hxp) # dimension = [batch size ,1 ,600]

        outputxm, _ = tf.nn.bidirectional_dynamic_rnn(self.cellx_fw, self.cellx_bw, \
        self.Xm_placeholder, initial_state_fw=_state_fw, initial_state_bw=_state_bw, dtype=tf.float64)
        hxm = tf.concat(outputxm, axis=2)
        outxm = self.attention_layer(hxm)

        self.distp = tf.reduce_sum(tf.abs(tf.subtract(outx, outxp)),axis=2) # dimension = [batch size,1]
        self.distm = tf.reduce_sum(tf.abs(tf.subtract(outx, outxm)),axis=2) # dimension = [batch size,1]

    def validate(self):

        self.__build_lstm()
        self.__prepare_files()

        zero_state = np.zeros((2,self.batch_size, self.state_size))
        _statex_fw = zero_state
        _statex_bw = zero_state
        _statexp_fw = zero_state
        _statexp_bw = zero_state
        _statexm_fw = zero_state
        _statexm_bw = zero_state

        with tf.Session() as sess :
            tf.global_variables_initializer().run()
            __x_batch = np.zeros((1,self.max_sentence_length,self.num_of_features))
            __xp_batch = np.zeros((1,self.max_sentence_length,self.num_of_features))
            __xm_batch = np.zeros((1,self.max_sentence_length,self.num_of_features))

            loss,_, _statex_fw, _statex_bw, _statexp_fw, _statexp_bw, \
            _statexm_fw, _statexm_bw = \
            sess.run([self.loss, self.optimizer, self.statex_fw, self.statex_bw,\
            self.statexp_fw, self.statexp_bw, self.statexm_fw, self.statexm_bw],\
            feed_dict={self.batchX_placeholder : __x_batch,\
            self.batchXp_placeholder : __xp_batch,\
            self.batchXm_placeholder : __xm_batch,\
            self.statex_fw_placeholder : _statex_fw,\
            self.statex_bw_placeholder : _statex_bw,\
            self.statexp_fw_placeholder : _statexp_fw, \
            self.statexp_bw_placeholder : _statexp_bw,\
            self.statexm_fw_placeholder : _statexm_fw, \
            self.statexm_bw_placeholder : _statexm_bw})

        dir_path = lstm_config.NEW_LSTM_FILE_STACK
        #epochs = sorted(os.listdir(dir_path))
        epochs = []
        for i in range(self.num_of_epochs):
            epochs.append('epoch_' + str(i))
        #data_files = sorted(os.listdir(dir_path + epochs[0]))
        data_files = ['model_10240.pickle']

        self.W = tf.get_variable('weight', initializer = np.zeros((self.attention_length,2*self.state_size)), dtype=tf.float64)
        self.b = tf.get_variable('bias', initializer = np.zeros((self.attention_length,self.max_sentence_length)), dtype=tf.float64)
        self.Uw = tf.get_variable('external_weight', initializer = np.zeros((self.attention_length,1)), dtype=tf.float64)
        try:
            with open(lstm_config.VALIDATION_TEMP_FILE,'rb') as f :
                [last_epoch_num,last_file_num, last_data_row] = pickle.load(f)

            if last_file_num == len(data_files) - 1 :
                last_epoch_num += 1
                last_file_num = 0
                last_data_row = []
        except FileNotFoundError :
            last_epoch_num = 0
            last_file_num = 0
            last_data_row = []
            with open(lstm_config.VALIDATION_REPORT,'a') as csvfile :
                csv_report = csv.writer(csvfile)
                csv_report.writerow(['epochs'] + data_files)

        widgets = ['TF Validation : ', progressbar.Percentage(), progressbar.Bar(),progressbar.ETA()]
        bar = progressbar.ProgressBar(widgets=widgets, maxval= len(epochs) * len(data_files)).start()

        for epoch_num in range(last_epoch_num, len(epochs)) :
            data_row = last_data_row
            last_data_row = []
            for file_num in range(last_file_num, len(data_files)) :
                with open(dir_path + epochs[epoch_num] + '/' + data_files[file_num], 'rb') as f:
                    [_,_,_,_, external_weight, weight_attn, bias_attn,weightf, weightb, _,_,_,_,_,_,_,_] = pickle.load(f)

                tf.assign(self.W, weight_attn)
                tf.assign(self.b, bias_attn)
                tf.assign(self.Uw, external_weight)
                self.cellx_fw.set_weights(weightf)
                self.cellx_bw.set_weights(weightb)

                _state_fw, _state_bw = self.__lstm_state_tuple(zero_state, zero_state)
                self.__build_test()

                self.__end_of_data = False
                with tf.Session() as sess : # config=tf.ConfigProto(intra_op_parallelism_threads=10)
                    tf.global_variables_initializer().run()
                    while not self.__end_of_data :
                        self.__gen_validation_batches()

                        for x, xp, xm in zip(self.__x_batch, self.__xp_batch, self.__xm_batch) :

                            distp,distm = sess.run([self.distp, self.distm], feed_dict={self.X_placeholder:[x],\
                            self.Xp_placeholder:[xp], self.Xm_placeholder:[xm]})

                            if distm >= distp :
                                self.pass_count += 1
                            else:
                                self.fail_count += 1

                pass_percent = (self.pass_count/(self.pass_count + self.fail_count))*100
                data_row.append(pass_percent)
                self.pass_count = 0
                self.fail_count = 0
                bar.update((epoch_num * len(data_files)) + (file_num + 1))
                logging.info('epoch : {}, data file : {}, data : {} '.format(epochs[epoch_num],data_files[file_num], data_row))
                with open(lstm_config.VALIDATION_TEMP_FILE,'wb') as f :
                    pickle.dump([epoch_num,file_num,data_row],f)

            with open(lstm_config.VALIDATION_REPORT,'a') as csvfile :
                csv_report = csv.writer(csvfile)
                csv_report.writerow([epochs[epoch_num]] + data_row)

        bar.finish()
        os.system('rm -rf /tmp/validation_metadata.pickle')
        pid = subprocess.check_output('ps ax | grep validate.sh' ,shell=True).split()[0].decode('utf-8')
        os.system('kill ' + pid)

valid = Validate_LSTM()
valid.validate()
