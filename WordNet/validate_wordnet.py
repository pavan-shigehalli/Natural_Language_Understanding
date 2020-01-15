""" Validates the WordNet based model with triplets extracted from Wikipdia """

import os
import csv
import math
import time
import logging
import pickle
import sys
import subprocess
import re
import requests

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
import numpy as np
import progressbar

from config import Glove as glove_config
from config import WordNet as wn_config

logging.basicConfig(filename=wn_config.LOG_FILE, format='[ %(asctime)s ] %(message)s',\
level=logging.INFO)

class Validate() :
    ''' validates the wordnet performance for sentence triplet relative
    distance measurements '''

    def __init__(self,validation_limit=None):
        self.validation_limit = validation_limit
        self.__valid_count = 0
        self.sen_limit_flag = False
        self.title_limit_flag = False
        self.end_of_data = False

        self.pass_count = { 'path' : 0, 'wup' : 0, 'lch' :0, 'UMBC': 0 }

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
            with open(wn_config.TMP_SAVE_FILE) as file :
                self.restore_file_exists = True
        except FileNotFoundError :
                self.restore_file_exists = False
                with open(wn_config.REPORT_FILE,'a') as csvfile :
                    csv_report = csv.writer(csvfile)
                    csv_report.writerow(['Triplet count', 'Path sim ', 'WUP sim', 'LCH sim', 'UMBC sim'])


    def penn_to_wn(self,tag):
        """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
        if tag.startswith('N'):
            return 'n'

        if tag.startswith('V'):
            return 'v'

        if tag.startswith('J'):
            return 'a'

        if tag.startswith('R'):
            return 'r'

        return None


    def tagged_to_synset(self,word, tag):
        wn_tag = self.penn_to_wn(tag)
        if wn_tag is None:
            return None

        try:
            return wn.synsets(word, wn_tag)[0]
        except:
            return None


    def gen_synset(self, sentence1, sentence2) :
        sentence1 = pos_tag(word_tokenize(sentence1))
        sentence2 = pos_tag(word_tokenize(sentence2))

        # Get the synsets for the tagged words
        synsets1 = [self.tagged_to_synset(*tagged_word) for tagged_word in sentence1]
        synsets2 = [self.tagged_to_synset(*tagged_word) for tagged_word in sentence2]

        # Filter out the Nones
        synsets1 = [ss for ss in synsets1 if ss]
        synsets2 = [ss for ss in synsets2 if ss]

        return synsets1, synsets2


    def __compute_score(self,synsets1, synsets2, similarity_measure) :
        score, count = 0.0, 0
        # For each word in the first sentence
        for synset in synsets1:
            # Get the similarity value of the most similar word in the other sentence
            scores = []
            similarity = None
            for ss in synsets2 :

                if similarity_measure == 'path_similarity' :
                    similarity = synset.path_similarity(ss)

                elif similarity_measure == 'lch_similarity' :
                    if synset.pos() == ss.pos() :
                        similarity = synset.lch_similarity(ss)

                elif similarity_measure == 'wup_similarity' :
                    similarity = synset.wup_similarity(ss)

                elif similarity_measure == 'res_similarity' : # not working
                    similarity = synset.res_similarity(ss)

                elif similarity_measure == 'jcn_similarity' : # not working
                    similarity = synset.jcn_similarity(ss)

                else:
                    raise KeyError("Undefined similarity measrure")

                if similarity :
                    scores.append(similarity)

            if scores :
                best_score = max(scores)
            else:
                best_score = None

            # Check that the similarity could have been computed
            if best_score is not None:
                score += best_score
                count += 1
        # Average the values
        try:
            score /= count
        except ZeroDivisionError :
            score = 0

        return score


    def compute_score(self,synsets1, synsets2, similarity_measure) :
        score1 = self.__compute_score(synsets1, synsets2, similarity_measure)
        score2 = self.__compute_score(synsets2, synsets1, similarity_measure)

        score = 0.5*score1 + 0.5*score2

        return score


    def sentence_similarity(self,sentence1, sentence2):
        """ compute the sentence similarity using Wordnet """

        synsets1, synsets2 = self.gen_synset(sentence1, sentence2)

        score_path = self.compute_score(synsets1, synsets2, 'path_similarity')
        score_wup = self.compute_score(synsets1, synsets2, 'wup_similarity')
        score_lch = self.compute_score(synsets1, synsets2, 'lch_similarity')

        return score_path, score_wup, score_lch


    def UMBC_STS(self, sentence1, sentence2):

        url = "http://swoogle.umbc.edu/StsService/GetStsSim?operation=api"

        # convert sentence to HTML format
        sentence1 = re.sub(' ','%20',re.sub(r"[^a-zA-Z0-9]+", ' ', sentence1))
        sentence2 = re.sub(' ','%20',re.sub(r"[^a-zA-Z0-9]+", ' ', sentence2))

        url += "&phrase1=" + sentence1 + "&phrase2=" + sentence2

        response = requests.get(url)
        time.sleep(0.2)
        score = float(response.text.strip())

        return score


    def gen_validation_batches(self) :
        __x_batch = []
        __xp_batch = []
        __xm_batch = []

        if not self.sen_limit_flag:
            for line_count,row in enumerate(self.sen_file) :
                __x_batch = row[0]
                __xp_batch = row[1]
                __xm_batch = row[2]
                self.__valid_count += 1
                break
            if self.validation_sen_limit == self.__valid_count :
                self.sen_limit_flag = True
                self.__valid_count = 0
        else :
            for line_count,row in enumerate(self.title_file) :
                __x_batch = row[0]
                __xp_batch = row[1]
                __xm_batch = row[2]
                self.__valid_count += 1
                break
            if self.validation_title_limit == self.__valid_count :
                self.sen_limit_flag = False
                self.end_of_data = True
                self.__valid_count = 0

        return __x_batch, __xp_batch, __xm_batch


    def validate(self) :

        if self.restore_file_exists :
            with open(wn_config.TMP_SAVE_FILE,'rb') as file :
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

            score_path_p, score_wup_p, score_lch_p = self.sentence_similarity(x_batch,xp_batch)
            score_path_m, score_wup_m, score_lch_m = self.sentence_similarity(x_batch, xm_batch)


            if score_path_p > score_path_m :
                self.pass_count['path'] += 1

            if score_wup_p > score_wup_m :
                self.pass_count['wup'] += 1

            if score_lch_p > score_lch_m :
                self.pass_count['lch'] += 1

            score_p = self.UMBC_STS(x_batch, xp_batch)
            score_m = self.UMBC_STS(x_batch, xm_batch)

            if score_p > score_m :
                self.pass_count['UMBC'] += 1

            triplet_count += 1

            if triplet_count % 100 == 0 :
                with open(wn_config.REPORT_FILE,'a') as csvfile :
                    csv_report = csv.writer(csvfile)
                    csv_report.writerow([triplet_count, self.pass_count['path'], self.pass_count['wup'], self.pass_count['lch'], self.pass_count['UMBC']])
                with open(wn_config.TMP_SAVE_FILE,'wb') as file :
                    pickle.dump([self.__valid_count, self.sen_limit_flag, triplet_count,self.pass_count],file)
                logging.info('saving with a triplet count : {}'.format(triplet_count))

            if triplet_count > self.validation_limit :
                break

            bar.update(triplet_count)
        bar.finish()

        pid = subprocess.check_output('ps ax | grep validate_wn.sh' ,shell=True).split()[0].decode('utf-8')
        os.system('kill ' + pid)


if __name__ == '__main__' :
    valid = Validate(validation_limit=1000)

    valid.validate()
