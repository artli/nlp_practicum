# Toxic Comment Classification Flask App

## Project Setup

### Install dependencies

```shell
eval "$(pyenv init -)"
pyenv install $(cat runtime.txt)
pip install --user pipenv
pipenv install
pipenv shell
```

### Train the model

1. Download the train data at https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data.
2. Download the embedding to `./assets/embedding/fasttext-crawl-300d-2m/crawl-300d-2M.vec~`:

```python
python src/download.py
```

3. Train the model (a pooled GRU with FastText embedding), serialising and storing it (at `./assets/model/model.h5`) together with the preprocessor (at `./assets/model/preprocessor.pkl`):
```python
python src/train_classifier.py
```

### Get the predictions

Run a test batch:
```python
python src/predict.py

# output:
# Corgi is stupid          - Toxicity: [0.99293655]
# good boy                 - Toxicity: [0.02075008]
# School of AI is awesome  - Toxicity: [0.01223523]
# F**K                     - Toxicity: [0.90747666]
```

## Resources
1. [ToxicFlaskApp](https://github.com/xinbinhuang/ToxicFlaskApp): a baseline from Vancouver School of AI