''' This is a configuration file and it contains absolute paths of files
 required by the other programs '''

import os
from os.path import dirname


class LSTM():
    # LSTM triplet network saver configuratios.
    parent_dir = os.getcwd()
    parent_dir += '/Natural_Language_Understanding'

    MODEL_DIR = parent_dir + '/Trained_Model/IBM_Debater/'
    SAVED_LSTM = MODEL_DIR + 'model.ckpt'
    NEW_LSTM = MODEL_DIR + 'model.ckpt'
    NEW_LSTM_FILE_STACK = MODEL_DIR + 'datastack/'
    TRAIN_LOG = parent_dir + '/IBM_Debater/logs/train.log'
    VALIDATION_REPORT = NEW_LSTM_FILE_STACK + 'validation_report.csv'
    REPORT_LOG = parent_dir + '/IBM_Debater/logs/validate.log'
    VALIDATION_TEMP_FILE = parent_dir + '/IBM_Debater/tmp/validation_temp.pickle'
