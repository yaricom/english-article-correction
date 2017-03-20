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

Before running experiments unpack data corpus archive file `data.zip` into `data` directory

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
* [evaluate.py](src/evaluate.py) - the results evaluation script where the target metric is not accuracy but recall level at a specified false positive rate level.

## Running experiments

Before running experiments with training a model that detects and corrects an incorrect usage of 
the English article (“a”, “an” or “the”) we need to process raw data corpora files.

To generate data set files execute from the terminal in the root directory following command:
```
$ create_data_set.sh
```

The generated data sets comprise features and labels per specific text corpora. The total of sixteen features considered as important.
Some of the features represented the words and POS tags found at specific locations adjacent to the determiner (only English articles: a, an, the); 
others represented the nouns, and verbs that preceded or followed the preposition.
Table 1 shows a subset of the feature list.

| Index | Feature | Description |
| ----- | ------- | ----------- |
| 0     | PrW     | The preceding word's Glove index|
| 1     | PrW POS | The preceding word's POS tag |
| 2     | DTa     | The Glove index of determiner (only English articles: a, an, the) |
| 3     | FlW     | The following word's Glove index |
| 4     | FlW POS | The following word's POS tag |
| 5     | FlW2    | The second following word's Glove index |
| 6     | FlW2 POS | The second following word's POS tag |
| 7     | FlNNs (i > 0) | The Glove index of following single/plural noun (where DT is not at the start of sentence) |
| 8     | FlNNs (i > 0) POS | The following single/plural noun's POS tag |
| 9     | PrW2 | The second preceding word's Glove index |
| 10    | PrW2 POS | The second preceding word's POS tag |
| 11    | PrW3 (VB, VBD) | The Glove index of third preceding verb (VB, VBD) if present |
| 12    | PrW3 (VB, VBD) POS | The POS tag of third preceding verb (VB, VBD) if present |
| 13    | Vowel [0, 1] | The flag to indicate if following word starts with specific vowel |
| 14    | FlW (VB,VBD,VBG,VBN,VBP,VBZ) | The Glove index of third following verb (VB,VBD,VBG,VBN,VBP,VBZ) if present |
| 15    | FlW (VB,VBD,VBG,VBN,VBP,VBZ) POS | The POS tag of third following verb (VB,VBD,VBG,VBN,VBP,VBZ) if present |

**Table 1** The features list description

To run prediction against *validation corpora* with subsequent evaluation against ground truth 
labels execute from the terminal in the root directory following command:
```
$ run_validate.sh
```

To run prediction against *test corpora* with default parameters execute from the terminal 
in the root directory following command:
```
$ generate_results.sh
```

The predicted results will be saved in `out` directory as `submission_test.txt` file.

## Conclusions

As result of provided experiment and analysis several conclusions can be made:
1. The provided data corpora has small number of samples which excludes building of advanced predictive models
based on neural network methods.
2. The best predictive performance was achieved with ensemble classifiers based on decision trees architecture.
3. Among tried decision tree algorithms the best prediction score was achieved with 
[Random Forest Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
4. It was found that optimal number of estimators for classifier is 15000. Other parameters was selected as by default.

The results for validation corpora by running `evaluate.py` script:
```
FP counts: {'the': 1519, 'an': 77,  'a': 435} 
FN counts: {'the': 1054, 'an': 112, 'a': 597}
TP counts: {'the': 5114, 'an': 97,  'a': 1142}
TN counts: {'the': 7093, 'an': 286, 'a': 2084}
>>> target score = 40.28 % (measured with 0.02 false positive rate level)
>>> accuracy (just for info) = 80.70 %
```

## Authors

This source code maintained and managed by Iaroslav Omelianenko





