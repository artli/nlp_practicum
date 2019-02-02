import os
from textgenrnn import textgenrnn

from .utils import get_root


DIR_ROOT = get_root()
DIR_ASSETS = os.path.join(DIR_ROOT, "assets")
DATA_DIR = os.path.join(DIR_ASSETS, "data")
MODEL_PATH = os.path.join(DIR_ROOT, "assets", "model")
WEIGHTS_FILE = os.path.join(MODEL_PATH, "reddit_legaladvice_relationshipadvice.hdf5")

model_name = "biased_generator"
model_cfg = {
    "word_level": True,  # set to True if want to train a word-level model (requires more data and smaller max_length)
    "rnn_size": 128,  # number of LSTM cells of each layer (128/256 recommended)
    "rnn_layers": 3,  # number of LSTM layers (>=2 recommended)
    "rnn_bidirectional": False,  # consider text both forwards and backward, can give a training boost
    "max_length": 5,  # number of tokens to consider before predicting the next (20-40 for characters, 5-10 for words recommended)
    "max_words": 10000,  # maximum number of words to model; the rest will be ignored (word-level model only)
}

train_cfg = {
    "line_delimited": True,  # set to True if each text has its own line in the source file
    "num_epochs": 20,  # set higher to train the model for longer
    "gen_epochs": 1,  # generates sample text from model after given number of epochs
    "train_size": 0.8,  # proportion of input data to train on: setting < 1.0 limits model from learning perfectly
    "dropout": 0.0,  # ignore a random proportion of source tokens each epoch, allowing model to generalize better
    "validation": False,  # If train__size < 1.0, test on holdout dataset; will make overall training slower
    "is_csv": True,  # set to True if file is a CSV exported from Excel/BigQuery/pandas
}

textgen = textgenrnn(name=model_name, weights_path=WEIGHTS_FILE)


def generate_using_type(class_type, split):
    train_function = (
        textgen.train_from_file
        if train_cfg["line_delimited"]
        else textgen.train_from_largetext_file
    )
    data_path = os.path.join(DATA_DIR, f"{class_type}_1.{split}.csv")
    train_function(
        file_path=data_path,
        is_csv=train_cfg["is_csv"],
    )


if __name__ == '__main__':
   generate_using_type("toxic", "combined")
