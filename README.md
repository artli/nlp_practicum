## Setup for training

### Install dependencies
```shell
eval "$(pyenv init -)"
pyenv install $(cat runtime.txt)
pyenv local $(cat runtime.txt)
pipenv install
pipenv check
```

### Download the data

Put the data from [the Kaggle repo](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) into `toxicity_analyzer/assets/data/`.

### Download the embedding
```shell
PYTHONPATH=. python -m toxicity_analyzer.model.download
```

## Run the web app

```shell
PYTHONPATH=. python -m toxicity_analyzer.app
```
## Resources
1. [ToxicFlaskApp](https://github.com/xinbinhuang/ToxicFlaskApp): a baseline from Vancouver School of AI
