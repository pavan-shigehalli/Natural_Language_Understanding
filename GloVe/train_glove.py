from .tf_glove import GloveEmbeddings
from .__init__ import Glove as glove_config
from ..Training_Data.__init__ import Triplet as triplet_config


def main() :
    train_file = triplet_config.WIKI_DATA # file for training
    saved_model = glove_config.SAVED_GLOVE_MODEL # Trained model
    new_model = glove_config.NEW_GLOVE_MODEL
    loss_report = glove_config.LOSS_REPORT

    Em = GloveEmbeddings(\
    train_file=train_file, \
    saved_model=None, \
    embedding_size=300, context_size=10,\
    loss_report_file=loss_report)

    Em.tf_train(num_epochs=1000, save_model= new_model)
    #Em.load_saved_model()


if __name__ == '__main__' :
    main()
