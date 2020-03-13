''' This is a configuration file and it contains absolute paths of files
 required by the other programs '''

import os
from os.path import dirname


class Glove():

    current_dir = os.getcwd()
    current_dir += '/Natural_Language_Understanding'
    
    MODEL_DIR = current_dir + '/Trained_Model/GloVe'
    SAVED_GLOVE_MODEL = MODEL_DIR + '/model.ckpt'
    NEW_GLOVE_MODEL = MODEL_DIR + '/model.ckpt'

    LOSS_REPORT = MODEL_DIR + '/loss.csv'
    VALIDATION_SAVE_FILE = current_dir + '/GloVe/tmp/validate_glove.pickle'
    SEN_LOG_FILE = current_dir + '/GloVe/logs/validate_glove_sen_embed.log'
    WORD_LOG_FILE = current_dir + '/GloVe/logs/validate_glove_word_embed.log'
    REPORT_FILE = MODEL_DIR + '/validate_report.csv'
