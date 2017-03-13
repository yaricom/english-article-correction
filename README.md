The repository contains the source code for experiment with applying NLP to correction of English definite/indefinite articles in text corpus

## Setting up working environment

### Dependencies

The source code in this directory is written in Python 3 and number of dependencies on
common Python libraries for ML and NLP:

1. [Numpy](https://docs.scipy.org/doc/numpy/index.html) - the Base N-dimensional array package
2. [Pandas](http://pandas.pydata.org/pandas-docs/stable/index.html) - the data analysis toolkit for Python
3. [scikit-learn](http://scikit-learn.org/stable/) - the collection of tools for data mining and data analysis

All dependencies can be installed manually or as part of distributive by installing `Anaconda` data science platform.
For `Anaconda` installation instructions visit [Anaconda web site](https://www.continuum.io/downloads). 
In order to use source code present in this directory install `Anaconda3` which includes `Python3`.

## The data corpus

The data corpora in this experiment was generated using UMBC WebBase corpus.
UMBC WebBase corpus by Lushan Han, UMBC Ebiquity Lab (http://ebiquity.umbc.edu/) is licensed under a 
Creative Commons Attribution 3.0 Unported License (http://creativecommons.org/licenses/by/3.0/deed.en_US). 
Based on a work at http://ebiq.org/r/351.

The data corpora comprise of following files:

1. sentences: sentence_train.txt, sentence_test.txt
2. corrections: corrections_train.txt, corrections_test.txt
3. constituency parse trees: parse_train.txt, parse_test.txt
4. dependency parse trees: dependencies_train.txt, dependencies_test.txt
5. part of speech tags: pos_tags_train.txt, pos_tags_test.txt
6. GloVe vectors indexes for list of vectors in glove_vectors.txt file: glove_train.txt, glove_test.txt

Before running experiments unpack data corpus archive file `data.zip` into `data` diretory


## Source code structure

The source code consists of series of `Python3` scripts encapsulating specific functionality:

* [config.py](src/config.py) - holds common configuration parameters (file and directories names, global constants, etc)
* [data_set.py](src/data_set.py) - the data sets generator to transform raw corpora into Numpy arrays with features and labels
* [data_set_test.py](src/data_set_test.py) - the unit tests for `data_set.py` script
* [tree_dict.py](src/tree_dict.py) - the parser and manipulation utilities for `constituency parse trees`
* [tree_dict_test.py](src/tree_dict_test.py) - the unit tests for `tree_dict.py` script
* [utils.py](src/utils.py) - the common utilities, such as: JSON parsing, data corpora sanity checks, etc
* [predictor.py](src/predictor.py) - the predictive models runner. Encapsulates common functionality which can be applied
to different predictors.
* [random_forest_model.py](src/random_forest_model.py) - the predictive model based on `sklearn.ensemble.RandomForestClassifier`

## Running experiments

Before running experiments with training a model that detects and corrects an incorrect usage of 
the English article (“a”, “an” or “the”) we need to process raw data corpora files.

To generate data set files execute from terminal in root directory the command:
```
$ create_data_set.sh
```

To run prediction with default parameters execute from terminal in root directory the command:
```
$ generate_results.sh
```

The predicted results will be saved in `out` directory as `submission_test.txt` file.

## Authors

This source code maintained and managed by Iaroslav Omelianenko





