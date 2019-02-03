import os
from random import uniform
from textgenrnn import textgenrnn

from .utils import get_root
from toxicity_analyzer import config


DIR_ROOT = get_root()
DIR_ASSETS = os.path.join(DIR_ROOT, "assets")
DATA_DIR = os.path.join(DIR_ASSETS, "data")
MODEL_PATH = os.path.join(DIR_ROOT, "assets", "model")


class Generator:
    def __init__(self, class_type, split_type="combined"):
        basename = f"{class_type}_1.{split_type}"
        weights_path = os.path.join(MODEL_PATH, f"{basename}.hd5")
        self._textgen = textgenrnn(name=basename, weights_path=weights_path)

    def generate(self, temperature=uniform(0.1, 0.5)):
        return self._textgen.generate(
            n=1, return_as_list=True, temperature=temperature
        )[0]

