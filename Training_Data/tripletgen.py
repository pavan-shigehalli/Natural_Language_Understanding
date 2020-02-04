''' Program to generate triplets from the file dataset.csv given by
 IBM debater project'''
import csv
from copy import copy
import random
import os
import operator as op
from functools import reduce

import progressbar

from config import Glove as config
from config import Triplet as triplet_config

class TripletGen():

    def __init__(self, infile, outfile_dir, max_out_file_size=1000000000000) :
        self.datafile = infile
        self.outfile_dir = outfile_dir
        self.max_out_file_size = max_out_file_size # 10 MB

        self.__sen_triplet_count = 0
        self.__title_triplet_count = 0
        self.__max_sen_triplet = 0
        self.__max_title_triplet = 0
        self.__sen_filecount = 0
        self.__title_filecount = 0
        self.__out_file = 'triplet'
        self.__logfile = 'triplet_count.csv'
        self.__current_dir = ''

        self.__max_triplet_count_init()


    def __max_triplet_count_init(self):
        ''' Counts the maximum number of possible triplet sentences and
        triplet titles in a csv file '''

        sentence_count = 0
        article = '' # Title of the article
        section = '' # article section
        prv_sec = False

        csvfile = open(self.datafile)
        csv_reader = csv.reader(csvfile)

        for i,row in enumerate(csv_reader) :
            if i == 0:
                continue

            if article != row[0] : # new article
                if article and section :
                    self.__max_sen_triplet += self.__ncr(sentence_count,2)
                    self.__max_title_triplet += sentence_count
                sentence_count = 0
                article = row[0]
                prv_sec = False

            if section != row[2] : # new section
                section = row[2]
                if sentence_count != 0 :
                    if prv_sec :
                        self.__max_sen_triplet += 2*self.__ncr(sentence_count,2)
                        self.__max_title_triplet += 2*sentence_count
                    else:
                        self.__max_sen_triplet += self.__ncr(sentence_count,2)
                        self.__max_title_triplet += sentence_count
                        prv_sec = True
                sentence_count = 0

            sentence_count += 1

        self.__max_sen_triplet += self.__ncr(sentence_count,2)
        self.__max_title_triplet += sentence_count


    def __ncr(self, n, r):
        ''' computes combinatorics '''
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return int(numer / denom)


    def __log_writer_init(self):
        ''' Initiates the log file '''
        file = open(self.__current_dir + self.__logfile, 'w')
        file_writer = csv.writer(file)
        file_writer.writerow(['Triplet category', 'triplet count' ])
        file.close()


    def __get_dir(self, triplet_count) :
        ''' Creates the directory for storing triplets and Returns
        the name of the directory '''

        if triplet_count >= self.__trainset[0] and \
        triplet_count <= self.__trainset[1] :
            dir = 'train/'
        elif triplet_count >= self.__validation_set[0] and \
        triplet_count <= self.__validation_set[1] :
            dir = 'validation/'
        elif triplet_count >= self.__test_set[0] and \
        triplet_count <= self.__test_set[1] :
            dir = 'test/'
        else:
            dir = ''

        try :
            os.mkdir(self.__current_dir + dir)

            if dir == 'validation/' :
                file = open(self.__current_dir + '/' + self.__logfile, 'a')
                file_writer = csv.writer(file)
                file_writer.writerow(['train', triplet_count])
                self.__true_train_count = triplet_count
                file.close()
            elif dir == 'test/' :
                file = open(self.__current_dir + '/' + self.__logfile, 'a')
                file_writer = csv.writer(file)
                file_writer.writerow(['validation', triplet_count - self.__true_train_count])
                self.__true_validation_count = triplet_count - self.__true_train_count
                file.close()
        except OSError: # If the directory already exists
            pass

        return dir

    def __write_triplet_title(self,article, section_list, sentence_list) :
        ''' forms triplets in the order x, x+, x- '''

        dir = self.__get_dir(self.__title_triplet_count)

        file = self.__current_dir + dir + self.__out_file + '_' + str(self.__title_filecount) + '.csv'
        try :
            if os.path.getsize(file) >= self.max_out_file_size :
                self.__title_filecount += 1
                print('Triplet count : {}'.format(self.__title_triplet_count))
                file = self.__current_dir + dir + self.__out_file + '_' + str(self.__title_filecount) + '.csv'
        except FileNotFoundError:
            pass

        triplet_title_writer = csv.writer(open(file, 'a' ))
        for i in range(len(sentence_list)) :
            for j in range(len(sentence_list[i])) :
                try :
                    if i < len(sentence_list) - 1 : # if next section exists
                        row = [sentence_list[i][j], article + ' ' + section_list[i],\
                         article + ' ' + section_list [i + 1]]
                        triplet_title_writer.writerow(row)
                        self.__title_triplet_count += 1
                    if i >= 1: # if previous section exists
                        row = [sentence_list[i][j], article + ' ' + section_list[i],\
                         article + ' ' + section_list [i - 1]]
                        triplet_title_writer.writerow(row)
                        self.__title_triplet_count += 1
                except IndexError:
                    return False
        return True


    def __write_triplet_sen(self,article, section_list, sentence_list) :
        ''' forms triplets in the order x, x+, x- '''

        dir = self.__get_dir(self.__sen_triplet_count)

        file = self.__current_dir + dir + self.__out_file + '_' + str(self.__sen_filecount) + '.csv'
        try :
            if os.path.getsize(file) >= self.max_out_file_size :
                self.__sen_filecount += 1
                print('Triplet count : {}'.format(self.__sen_triplet_count))
                file = self.__current_dir + dir + self.__out_file + '_' + str(self.__sen_filecount) + '.csv'
        except FileNotFoundError:
            pass

        triplet_sen_writer = csv.writer(open(file, 'a' ))
        for i in range(len(sentence_list)) :
            for j in range(len(sentence_list[i])) :
                for k in range(j + 1,len(sentence_list[i])) :
                    try :
                        if i < len(sentence_list) - 1 : # If next section exists
                            rand = random.randint(0,len(sentence_list[ i + 1 ]) - 1)
                            row  = [sentence_list[i][j], sentence_list[i][k], sentence_list[i + 1][rand]]
                            triplet_sen_writer.writerow(row)
                            self.__sen_triplet_count += 1
                        if i >= 1 : # If previous section exists
                            rand = random.randint(0,len(sentence_list[ i - 1 ]) - 1 )
                            row  = [sentence_list[i][j], sentence_list[i][k], sentence_list[i - 1][rand]]
                            triplet_sen_writer.writerow(row)
                            self.__sen_triplet_count += 1
                    except IndexError :
                        return False
        return True


    def generate_triplet_title(self, trainset, validation_set, test_set):

        self.__current_dir = self.outfile_dir + 'triplet_title/'
        try :
            os.mkdir(self.__current_dir)
        except OSError:
            pass

        self.__trainset = [0, int((trainset/100)*self.__max_title_triplet)]

        self.__validation_set = [self.__trainset[1] + 1 , \
        int((validation_set/100)*self.__max_title_triplet) + self.__trainset[1]]

        self.__test_set = [self.__validation_set[1] , self.__max_title_triplet]
        self.__log_writer_init()

        sentence = []
        sentence_list = []
        section_list = [] # section in an article
        article = '' # Title of the article
        section = '' # article section

        csvfile = open(self.datafile)
        csv_reader = csv.reader(csvfile)
        print('preparing file for triplet title generation ')
        for line_count, line in enumerate(csv_reader) :
            # count the number of lines
            pass
        csvfile.close()
        csvfile = open(self.datafile)
        csv_reader = csv.reader(csvfile)

        widgets = ['File formation : ', progressbar.Percentage(), progressbar.Bar(),\
         progressbar.ETA()]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=line_count).start()

        for i,row in enumerate(csv_reader) :
            if i == 0:
                continue

            if article != row[0] : # new article
                # form the triplets and write it to a file
                if article and section_list and sentence_list :
                    sentence_list.append(copy(sentence))
                    status = self.__write_triplet_title(article, section_list, sentence_list)
                    if not status:
                        print('Error has occured while forming the triplets')
                section_list.clear()
                sentence_list.clear()
                sentence.clear()
                article = row[0]
            if section != row[2] : # new section
                section = row[2]
                if sentence :
                    sentence_list.append(copy(sentence))
                section_list.append(section)
                sentence.clear()

            sentence.append(row[1])
            bar.update(i)

        sentence_list.append(copy(sentence))
        status = self.__write_triplet_title(article, section_list, sentence_list)

        bar.finish()
        log = csv.writer(open(self.__current_dir + '/' + self.__logfile, 'a'))
        log.writerow(['test', self.__title_triplet_count - self.__true_train_count - \
        self.__true_validation_count])
        log.writerow(['Total count', self.__title_triplet_count])
        log.writerow(['Total theoretical count', self.__max_title_triplet])
        print('{} number of triplet titles are formed'.format(self.__title_triplet_count))


    def generate_triplet_sen(self, trainset, validation_set, test_set):

        self.__current_dir = self.outfile_dir + 'triplet_sen/'
        os.system('mkdir -p ' + self.__current_dir)

        self.__trainset = [0, int((trainset/100)*self.__max_sen_triplet)]

        self.__validation_set = [self.__trainset[1] + 1 , \
        int((validation_set/100)*self.__max_sen_triplet) + self.__trainset[1]]

        self.__test_set = [self.__validation_set[1] , self.__max_sen_triplet]
        self.__log_writer_init()

        sentence = []
        sentence_list = []
        section_list = [] # section in an article
        article = '' # Title of the article
        section = '' # article section

        csvfile = open(self.datafile)
        csv_reader = csv.reader(csvfile)
        print('preparing file for triplet sentence generation')
        for line_count, line in enumerate(csv_reader) :
            # count the number of lines
            pass
        csvfile.close()
        csvfile = open(self.datafile)
        csv_reader = csv.reader(csvfile)

        widgets = ['File formation : ', progressbar.Percentage(), progressbar.Bar(),\
         progressbar.ETA()]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=line_count).start()

        for i,row in enumerate(csv_reader) :
            if i == 0:
                continue

            if article != row[0] : # new article
                # form the triplets and write it to a file
                if article and section_list and sentence_list :
                    sentence_list.append(copy(sentence))
                    status = self.__write_triplet_sen(article, section_list, sentence_list)
                    if not status:
                        print('Error has occured while forming the triplets')
                section_list.clear()
                sentence_list.clear()
                sentence.clear()
                article = row[0]
            if section != row[2] : # new section
                section = row[2]
                if sentence :
                    sentence_list.append(copy(sentence))
                section_list.append(section)
                sentence.clear()

            sentence.append(row[1])
            bar.update(i)

        sentence_list.append(copy(sentence))
        status = self.__write_triplet_sen(article, section_list, sentence_list)

        bar.finish()
        log = csv.writer(open(self.__current_dir + self.__logfile, 'a'))
        log.writerow(['test', self.__sen_triplet_count - self.__true_train_count - \
        self.__true_validation_count])
        log.writerow(['Total count', self.__sen_triplet_count])
        log.writerow(['Total theoretical count', self.__max_sen_triplet])
        print('{} number of triplet sentences are formed'.format(self.__sen_triplet_count))



def main():
    datafile = triplet_config.WIKI_DATA
    triplet_dir = triplet_config.TRIPLET_DIR
    trip = TripletGen(datafile, triplet_dir)
    trip.generate_triplet_sen(trainset=80, validation_set=10, test_set=10)
    trip.generate_triplet_title(trainset=80, validation_set=10, test_set=10)


if __name__ == '__main__':
    main()
