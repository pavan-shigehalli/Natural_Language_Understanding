''' This is a configuration file and it contains absolute paths of files
 required by the other programs '''

import os
from os.path import dirname

class Triplet():
    current_dir = os.getcwd()
    parent_dir = dirname(current_dir)

    LOG_FILE = parent_dir + '/Training_Data/triplets'
    WIKI_DATA = parent_dir + '/Training_Data/dataset.csv'
    # absolute path of the directory to store the generated triplets
    TRIPLET_DIR = current_dir + '/triplets/'
    TRIPLET_SEN_DIR = 'triplet_sen/'
    TRIPLET_TITLE_DIR = 'triplet_title/'
    TRIPLET_TRAIN_DIR = 'train/' # relative directory
    TRIPLET_VALID_DIR = 'validation/'
    TRIPLET_TEST_DIR = 'test/'
    TRIPLET_SEN_LOG = TRIPLET_DIR + TRIPLET_SEN_DIR + 'triplet_count.csv'
    TRIPLET_TITLE_LOG = TRIPLET_DIR + TRIPLET_TITLE_DIR + 'triplet_count.csv'


class Glove():

    parent_dir = dirname(os.getcwd())
    # Word embedding saver configuration
    GLOVE_TRAIN = parent_dir + '/Training_Data/dataset.csv'
    MODEL_DIR = parent_dir + '/Trained_Model/GloVe'
    SAVED_GLOVE_MODEL = MODEL_DIR + '/model.ckpt'
    NEW_GLOVE_MODEL = MODEL_DIR + '/model.ckpt'
    # absolute path of the directory to store the generated triplets
    TRIPLET_DIR = MODEL_DIR + '/triplets/'
    TRIPLET_SEN_DIR = 'triplet_sen/'
    TRIPLET_TITLE_DIR = 'triplet_title/'
    TRIPLET_TRAIN_DIR = 'train/' # relative directory
    TRIPLET_VALID_DIR = 'validation/'
    TRIPLET_TEST_DIR = 'test/'
    TRIPLET_SEN_LOG = TRIPLET_DIR + TRIPLET_SEN_DIR + 'triplet_count.csv'
    TRIPLET_TITLE_LOG = TRIPLET_DIR + TRIPLET_TITLE_DIR + 'triplet_count.csv'

    LOSS_REPORT = MODEL_DIR + '/loss.csv'
    VALIDATION_SAVE_FILE = parent_dir + '/GloVe/tmp/validate_glove.pickle'
    SEN_LOG_FILE = parent_dir + '/GloVe/logs/validate_glove_sen_embed.log'
    WORD_LOG_FILE = parent_dir + '/GloVe/logs/validate_glove_word_embed.log'
    #REPORT_FILE = current_dir + '/trained_model/validate_glove/validate_report.csv'
