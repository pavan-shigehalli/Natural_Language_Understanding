import tf_glove
from config import Glove as config


def main() :
    train_file = config.GLOVE_TRAIN # file for training
    saved_model = config.SAVED_GLOVE_MODEL # Trained model
    new_model = config.NEW_GLOVE_MODEL

    Em = GloveEmbeddings(\
    train_file=None, \
    saved_model=config.SAVED_GLOVE_MODEL, \
    embedding_size=300, context_size=10,\
    loss_report_file=config.LOSS_REPORT)

    #Em.tf_train(num_epochs=1000, save_model= config.NEW_GLOVE_MODEL)
    Em.load_saved_model()


if __name__ == '__main__' :
    main()
