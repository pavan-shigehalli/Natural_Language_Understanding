''' This is a configuration file and it contains absolute paths of files
 required by the other programs '''

import os
from os.path import dirname

class Triplet():
    current_dir = os.getcwd()
    current_dir += '/Natural_Language_Understanding'

    LOG_FILE = current_dir + '/Training_Data/triplets'
    WIKI_DATA = current_dir + '/Training_Data/dataset.csv'
    # absolute path of the directory to store the generated triplets
    TRIPLET_DIR = current_dir + '/Training_Data/triplets/'
    TRIPLET_SEN_DIR = 'triplet_sen/'
    TRIPLET_TITLE_DIR = 'triplet_title/'
    TRIPLET_TRAIN_DIR = 'train/' # relative directory
    TRIPLET_VALID_DIR = 'validation/'
    TRIPLET_TEST_DIR = 'test/'
    TRIPLET_SEN_LOG = TRIPLET_DIR + TRIPLET_SEN_DIR + 'triplet_count.csv'
    TRIPLET_TITLE_LOG = TRIPLET_DIR + TRIPLET_TITLE_DIR + 'triplet_count.csv'
