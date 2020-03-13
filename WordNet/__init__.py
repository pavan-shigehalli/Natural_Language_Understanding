''' This is a configuration file and it contains absolute paths of files
 required by the other programs '''

import os
from os.path import dirname


class WordNet():
    current_dir = os.getcwd()
    current_dir += '/Natural_Language_Understanding'

    LOG_FILE = current_dir + '/WordNet/logs/validate_wordnet.log'
    REPORT_FILE = current_dir + '/WordNet/logs/validate_report.csv'
    TMP_SAVE_FILE = current_dir + '/WordNet/tmp/validate_wordnet.pickle'
