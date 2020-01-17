""" Program to validate the triplets based on pure BERT encoding """

from threading import Thread
import os
import csv
import math
import time
import logging
import pickle
import sys
import subprocess

from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer
from bert_serving.client import BertClient
import numpy as np
from numpy import linalg as LA
import progressbar

from config import BERT as BERT_config
from config import Glove as glove_config

logging.basicConfig(filename=BERT_config.LOG_FILE, format='[ %(asctime)s ] %(message)s',\
level=logging.INFO)

class Validate():
    ''' validates the BERT performance for sentence triplet relative
    distance measurements '''

    def __init__(self,validation_limit=None): # Limit on the number of triplets to be validated
        self.validation_limit = validation_limit
        self.__valid_count = 0
        self.sen_limit_flag = False
        self.title_limit_flag = False
        self.end_of_data = False

        self.pass_count ={ 'L1' : 0, 'L2' : 0, 'cosine' :0 , 'min' : 0, 'cheb' : 0}

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

        try :
            with open(BERT_config.TMP_SAVE_FILE) as file :
                self.restore_file_exists = True
        except FileNotFoundError :
                self.restore_file_exists = False
                with open(BERT_config.REPORT_FILE,'a') as csvfile :
                    csv_report = csv.writer(csvfile)
                    csv_report.writerow(['Triplet count', 'L1 dist', 'L2 dist', 'Cosine dist', 'Minskowiski dist', 'Chebyshev dist'])


    def __sentence_to_vector(self, sentence):
        '''' Generates the single batch of data '''
        return self.bert_client.encode([sentence])


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


    def measure_L1_dist(self,vec1,vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        distance = np.sum(np.absolute(np.subtract(vec1,vec2)))

        return distance


    def measure_L2_dist(self,vec1,vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        distance = np.sum(np.square(np.subtract(vec1,vec2)))

        return distance


    def measure_angle(self, vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        angle = np.arccos(np.true_divide(np.matmul(vec1, vec2.transpose()),(LA.norm(vec1)*LA.norm(vec2))))
        angle = angle[0][0] *(180/np.pi)

        return angle


    def minkowski_distance(self, vec1, vec2, power=5) :
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        distance = np.power(np.sum(np.power(np.abs(np.subtract(vec1,vec2)),power)),1/power)

        return distance


    def chebyshev_distance(self, vec1, vec2) :
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        distance = max(np.abs(np.subtract(vec1,vec2))[0])

        return distance


    def validate(self) :
        self.bert_client = BertClient()

        if self.restore_file_exists :
            with open(BERT_config.TMP_SAVE_FILE,'rb') as file :
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

            distp = self.measure_angle(x_batch,xp_batch)
            distm = self.measure_angle(x_batch,xm_batch)
            if distp < distm :
                self.pass_count['cosine'] += 1

            distp = self.minkowski_distance(x_batch,xp_batch)
            distm = self.minkowski_distance(x_batch,xm_batch)
            if distp < distm :
                self.pass_count['min'] += 1

            distp = self.chebyshev_distance(x_batch,xp_batch)
            distm = self.chebyshev_distance(x_batch,xm_batch)
            if distp < distm :
                self.pass_count['cheb'] += 1


            triplet_count += 1

            if triplet_count % 100 == 0 :
                with open(BERT_config.REPORT_FILE,'a') as csvfile :
                    csv_report = csv.writer(csvfile)
                    csv_report.writerow([triplet_count, self.pass_count['L1'], self.pass_count['L2'],\
                                         self.pass_count['cosine'],self.pass_count['min'],self.pass_count['cheb']])
                with open(BERT_config.TMP_SAVE_FILE,'wb') as file :
                    pickle.dump([self.__valid_count, self.sen_limit_flag, triplet_count,self.pass_count],file)
                logging.info('saving with a triplet count : {}'.format(triplet_count))

            if triplet_count > self.validation_limit :
                break

            bar.update(triplet_count)
        bar.finish()

        pid = subprocess.check_output('ps ax | grep validate_bert.sh' ,shell=True).split()[0].decode('utf-8')
        os.system('kill ' + pid)



    '''def start_bert_server(self) :
        os.system('cd /tmp && bert-serving-start -model_dir ' + BERT_config.model_path)

    def stop_bert_server(self) :
        os.system('bert-serving-terminate -port 5555')'''

if __name__ == '__main__' :
    #Thread(target=valid.start_bert_server,daemon=True).start() # start the server in the background
    valid = Validate(validation_limit=100000)
    valid.validate()
    #valid.stop_bert_server()
 
