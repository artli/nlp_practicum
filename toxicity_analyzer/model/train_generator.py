import os
from textgenrnn import textgenrnn

from .utils import get_root
from toxicity_analyzer import config


DIR_ROOT = get_root()
DIR_ASSETS = os.path.join(DIR_ROOT, "assets")
DATA_DIR = os.path.join(DIR_ASSETS, "data")
MODEL_PATH = os.path.join(DIR_ROOT, "assets", "model")
WEIGHTS_PATH = os.path.join(MODEL_PATH, "reddit_legaladvice_relationshipadvice.hdf5")

model_cfg = {
    "word_level": True,  # set to True if want to train a word-level model (requires more data and smaller max_length)
    "rnn_size": 128,  # number of LSTM cells of each layer (128/256 recommended)
    "rnn_layers": 5,  # number of LSTM layers (>=2 recommended)
    "rnn_bidirectional": True,  # consider text both forwards and backward, can give a training boost
    "max_length": 10,  # number of tokens to consider before predicting the next (20-40 for characters, 5-10 for words recommended)
    "max_words": 10000,  # maximum number of words to model; the rest will be ignored (word-level model only)
}

train_cfg = {
    "line_delimited": True,  # set to True if each text has its own line in the source file
    "num_epochs": 1,  # set higher to train the model for longer
    "gen_epochs": 1,  # generates sample text from model after given number of epochs
    "train_size": 0.9,  # proportion of input data to train on: setting < 1.0 limits model from learning perfectly
    "dropout": 0.1,  # ignore a random proportion of source tokens each epoch, allowing model to generalize better
    "validation": True,  # If train__size < 1.0, test on holdout dataset; will make overall training slower
    "is_csv": True,  # set to True if file is a CSV exported from Excel/BigQuery/pandas
}

split_types = ["combined"]


def train_on_type(class_type, split_type):
    basename = f"{class_type}_1.{split_type}"
    weights_path = os.path.join(MODEL_PATH, f"{basename}.hd5")
    data_path = os.path.join(DATA_DIR, f"{basename}.csv")
    textgen = textgenrnn(name=basename, weights_path=WEIGHTS_PATH)

    train_function = (
        textgen.train_from_file
        if train_cfg["line_delimited"]
        else textgen.train_from_largetext_file
    )

    print(f"Starting to train on {data_path}.")
    if os.path.isfile(weights_path):
        print(f"Loading weights from {weights_path}...")
        textgen.load(weights_path)

    train_function(
        data_path,
        new_model=False,
        num_epochs=train_cfg["num_epochs"],
        gen_epochs=train_cfg["gen_epochs"],
        batch_size=1024,
        train_size=train_cfg["train_size"],
        dropout=train_cfg["dropout"],
        validation=train_cfg["validation"],
        is_csv=train_cfg["is_csv"],
        rnn_layers=model_cfg["rnn_layers"],
        rnn_size=model_cfg["rnn_size"],
        rnn_bidirectional=model_cfg["rnn_bidirectional"],
        max_length=model_cfg["max_length"],
        dim_embeddings=100,
        word_level=model_cfg["word_level"],
    )
    textgen.save(weights_path)


def main():
    for split_type in split_types:
        for class_type in config.classes:
            train_on_type(class_type, split_type)


if __name__ == "__main__":
    main()

