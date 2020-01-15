""" Program to validate the triplets based on pure GloVe word encoding """

from threading import Thread
import os
import csv
import math
import time
import logging
import pickle
import sys
import subprocess
import warnings

import numpy as np
from numpy import linalg as LA
import progressbar

from config import Glove as glove_config
import tf_glove

logging.basicConfig(filename=glove_config.WORD_LOG_FILE, format='[ %(asctime)s ] %(message)s',\
level=logging.INFO)

class Validate():
    ''' validates the GloVe word embedding performance for sentence triplet relative
    distance measurements '''

    def __init__(self,validation_limit=None,\ # Set a cap on the number of triplets to be validated
                max_sentence_length = 128,\ # Maximum number of words in a sentence
                embedding_size=300,\ # GloVe word embedding size
                context_size=10,\ # Size of the left and the right window
                saved_model=glove_config.SAVED_GLOVE_MODEL): # GLoVe trained model

        self.validation_limit = validation_limit
        self.__valid_count = 0
        self.sen_limit_flag = False
        self.title_limit_flag = False
        self.end_of_data = False
        self.max_sentence_length = max_sentence_length
        self.embedding_size = embedding_size

        self.pass_count ={ 'L1' : 0, 'L2' : 0 , 'min' : 0, 'cheb' : 0, 'cosine' : 0}

        self.sen_file = csv.reader(open(glove_config.TRIPLET_DIR + glove_config.TRIPLET_SEN_DIR + \
        glove_config.TRIPLET_VALID_DIR + 'triplet_0.csv' ))

        self.title_file = csv.reader(open(glove_config.TRIPLET_DIR + glove_config.TRIPLET_TITLE_DIR + \
        glove_config.TRIPLET_VALID_DIR + 'triplet_0.csv' ))

        with open(glove_config.TRIPLET_SEN_LOG) as f :
            next(csv.reader(f))
            next(csv.reader(f))
            triplet_sen_count = int(next(csv.reader(f))[1])
        with open(glove_config.TRIPLET_TITLE_LOG) as f :
            next(csv.reader(f))
            next(csv.reader(f))
            triplet_title_count = int(next(csv.reader(f))[1])
            max_validate_triplet_count = triplet_sen_count + triplet_title_count

            fact = triplet_sen_count/(max_validate_triplet_count)
            if self.validation_limit is not None :
                self.validation_sen_limit = math.ceil(self.validation_limit * fact)
                self.validation_title_limit = self.validation_limit - self.validation_sen_limit
                if self.validation_title_limit == 0:
                    self.validation_title_limit = 1
                    self.validation_sen_limit -= 1
            else:
                self.validation_sen_limit = triplet_sen_count
                self.validation_title_limit = triplet_title_count

        self.__embed = tf_glove.GloveEmbeddings(\
        train_file=None, \
        saved_model=saved_model, \
        embedding_size=embedding_size,\
        context_size=context_size,\
        )

        self.__embed.load_saved_model()

        try :
            with open(glove_config.VALIDATION_SAVE_FILE) as file :
                self.restore_file_exists = True
        except FileNotFoundError :
                self.restore_file_exists = False
                with open(glove_config.REPORT_FILE,'a') as csvfile :
                    csv_report = csv.writer(csvfile)
                    csv_report.writerow(['Triplet count', 'L1 dist', 'L2 dist', 'Minskowiski dist', 'Chebyshev dist', 'cosine dist'])


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

        #time_axis = self.__mean_vector(time_axis, word_count)

        return time_axis


    '''def __mean_vector(self, matrix, word_count) :
        mat = np.array(matrix)
        vec = np.true_divide(np.sum(mat,axis=0),word_count)

        return vec'''


    def gen_validation_batches(self) :
        __x_batch = []
        __xp_batch = []
        __xm_batch = []

        if not self.sen_limit_flag:
            for line_count,row in enumerate(self.sen_file) :
                __x_batch = self.__sentence_to_vector(row[0])
                __xp_batch = self.__sentence_to_vector(row[1])
                __xm_batch = self.__sentence_to_vector(row[2])
                self.__valid_count += 1
                break
            if self.validation_sen_limit == self.__valid_count :
                self.sen_limit_flag = True
                self.__valid_count = 0
        else :
            for line_count,row in enumerate(self.title_file) :
                __x_batch = self.__sentence_to_vector(row[0])
                __xp_batch = self.__sentence_to_vector(row[1])
                __xm_batch = self.__sentence_to_vector(row[2])
                self.__valid_count += 1
                break
            if self.validation_title_limit == self.__valid_count :
                self.sen_limit_flag = False
                self.end_of_data = True
                self.__valid_count = 0

        return __x_batch, __xp_batch, __xm_batch


    def measure_L1_dist(self,mat1,mat2):
        #vec1 = np.array(mat1)
        #vec2 = np.array(mat2)
        distance = np.sum(np.sum(np.absolute(np.subtract(mat1,mat2)),axis=1),axis=0)
        #distance = np.sum(np.absolute(np.subtract(vec1,vec2)))

        return distance


    def measure_L2_dist(self,mat1,mat2):
        #vec1 = np.array(mat1)
        #vec2 = np.array(mat2)
        distance = np.sqrt(np.sum(np.sum(np.square(np.subtract(mat1,mat2)),axis=1),axis=0))
        #distance = np.sum(np.square(np.subtract(vec1,vec2)))

        return distance


    '''def measure_angle(self, mat1, mat2):
        vec1 = np.array(mat1)
        vec2 = np.array(mat2)

        #for vec1, vec2 in zip(mat1, mat2) :
        #    angle = np.arccos(np.true_divide(np.matmul(vec1, vec2.transpose()),(LA.norm(vec1)*LA.norm(vec2))))
        #    angle = angle[0][0] *(180/np.pi)

        angle = np.arccos(np.true_divide(np.matmul(vec1, vec2.transpose()),(LA.norm(vec1)*LA.norm(vec2))))
        angle = angle*(180/np.pi)

        return angle'''


    def minkowski_distance(self, mat1, mat2, power=5) :
        #vec1 = np.array(mat1)
        #vec2 = np.array(mat2)

        distance = np.power(np.sum(np.sum(np.power(np.abs(np.subtract(mat1,mat2)),power), axis=1), axis=0),1/power)
        #distance = np.power(np.sum(np.power(np.abs(np.subtract(vec1,vec2)),power)),1/power)

        return distance


    def chebyshev_distance(self, mat1, mat2) :
        vec1 = np.array(mat1)
        vec2 = np.array(mat2)

        distance = np.abs(np.subtract(mat1,mat2)).max()
        #print(max(np.abs(np.subtract(vec1,vec2))))
        #distance = max(np.abs(np.subtract(vec1,vec2)))

        return distance


    def validate(self) :

        if self.restore_file_exists :
            with open(glove_config.VALIDATION_SAVE_FILE,'rb') as file :
                [self.__valid_count, self.sen_limit_flag, triplet_count,self.pass_count] = pickle.load(file)
            logging.info('RESTORING with a triplet count: {}'.format(triplet_count))

            # Restore to the last file status
            if not self.sen_limit_flag :
                for line_count,row in enumerate(self.sen_file) :
                    if line_count >= self.__valid_count :
                        break
                logging.info('restoring sentence file from the line : {}'.format(self.__valid_count))
            else:
                for line_count,row in enumerate(self.title_file) :
                    if line_count >= self.__valid_count :
                        break
                logging.info('restoring title file fromt the line : {}'.format(self.__valid_count))
        else:
            triplet_count = 0
            logging.info('STARTING fresh validation')

        widgets = ['Validation : ', progressbar.Percentage(), progressbar.Bar(),progressbar.ETA()]
        maxval = self.validation_sen_limit + self.validation_title_limit
        bar = progressbar.ProgressBar(widgets=widgets, maxval=maxval).start()

        while not self.end_of_data :
            x_batch, xp_batch, xm_batch = self.gen_validation_batches()

            distp = self.measure_L1_dist(x_batch,xp_batch)
            distm = self.measure_L1_dist(x_batch, xm_batch)
            if distp < distm :
                self.pass_count['L1'] += 1

            distp = self.measure_L2_dist(x_batch,xp_batch)
            distm = self.measure_L2_dist(x_batch, xm_batch)
            if distp < distm :
                self.pass_count['L2'] += 1

            distp = self.minkowski_distance(x_batch,xp_batch)
            distm = self.minkowski_distance(x_batch,xm_batch)
            if distp < distm :
                self.pass_count['min'] += 1

            distp = self.chebyshev_distance(x_batch,xp_batch)
            distm = self.chebyshev_distance(x_batch,xm_batch)
            if distp < distm :
                self.pass_count['cheb'] += 1

            '''distp = self.measure_angle(x_batch,xp_batch)
            distm = self.measure_angle(x_batch,xm_batch)
            if distp < distm :
                self.pass_count['cosine'] += 1'''


            triplet_count += 1

            if triplet_count % 100 == 0 :
                with open(glove_config.REPORT_FILE,'a') as csvfile :
                    csv_report = csv.writer(csvfile)
                    csv_report.writerow([triplet_count, self.pass_count['L1'], self.pass_count['L2'],\
                                         self.pass_count['min'],self.pass_count['cheb'],self.pass_count['cosine']])
                with open(glove_config.VALIDATION_SAVE_FILE,'wb') as file :
                    pickle.dump([self.__valid_count, self.sen_limit_flag, triplet_count,self.pass_count],file)
                logging.info('saving with a triplet count : {}'.format(triplet_count))

            if triplet_count > self.validation_limit :
                break

            bar.update(triplet_count)
        bar.finish()

        pid = subprocess.check_output('ps ax | grep validate_glove.sh' ,shell=True).split()[0].decode('utf-8')
        os.system('kill ' + pid)


if __name__ == '__main__' :

    valid = Validate(validation_limit=100000)
    valid.validate()
