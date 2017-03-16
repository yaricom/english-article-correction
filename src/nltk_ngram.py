#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:58:14 2017

@author: yaric
"""
import json
from math import log
import pickle

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer

from collections import Counter, defaultdict
from copy import copy
from itertools import chain

from nltk.util import ngrams
from nltk.probability import FreqDist, ConditionalFreqDist

import config

def build_vocabulary(cutoff, *texts):
    combined_texts = chain(*texts)
    return NgramModelVocabulary(cutoff, combined_texts)


def count_ngrams(order, vocabulary, *training_texts, **counter_kwargs):
    counter = NgramCounter(order, vocabulary, **counter_kwargs)
    for text in training_texts:
        counter.train_counts(text)
    return counter


class NgramModelVocabulary(Counter):
    """Stores language model vocabulary.
    Satisfies two common language modeling requirements for a vocabulary:
    - When checking membership and calculating its size, filters items by comparing
      their counts to a cutoff value.
    - Adds 1 to its size so as to account for "unknown" tokens.
    """

    def __init__(self, unknown_cutoff, *counter_args):
        Counter.__init__(self, *counter_args)
        self.cutoff = unknown_cutoff

    def __contains__(self, item):
        """Only consider items with counts GE to cutoff as being in the vocabulary."""
        return self[item] >= self.cutoff

    def __len__(self):
        """This should reflect a) filtering items by count, b) accounting for unknowns.
        The first is achieved by relying on the membership check implementation.
        The second is achieved by adding 1 to vocabulary size.
        """
        # the if-clause here looks a bit dumb, should we make it clearer?
        return sum(1 for item in self if item in self) + 1

    def __copy__(self):
        return self.__class__(self.cutoff, self)


class EmptyVocabularyError(Exception):
    pass


class NgramCounter(object):
    """Class for counting ngrams"""

    def __init__(self, order, vocabulary, unk_cutoff=None, unk_label="<UNK>", **ngrams_kwargs):
        """
        :type training_text: List[List[str]]
        """

        if order < 1:
            message = "Order of NgramCounter cannot be less than 1. Got: {0}"
            raise ValueError(message.format(order))

        self.order = order
        self.unk_label = unk_label

        # Preset some common defaults...
        self.ngrams_kwargs = {
            "pad_left": True,
            "pad_right": True,
            "left_pad_symbol": "<s>",
            "right_pad_symbol": "</s>"
        }
        # While allowing whatever the user passes to override them
        self.ngrams_kwargs.update(ngrams_kwargs)
        # Set up the vocabulary
        self._set_up_vocabulary(vocabulary, unk_cutoff)

        self.ngrams = defaultdict(ConditionalFreqDist)
        self.unigrams = FreqDist()

    def _set_up_vocabulary(self, vocabulary, unk_cutoff):
        self.vocabulary = copy(vocabulary)  # copy needed to prevent state sharing
        if unk_cutoff is not None:
            # If cutoff value is provided, override vocab's cutoff
            self.vocabulary.cutoff = unk_cutoff

        if self.ngrams_kwargs['pad_left']:
            lpad_sym = self.ngrams_kwargs.get("left_pad_symbol")
            self.vocabulary[lpad_sym] = self.vocabulary.cutoff

        if self.ngrams_kwargs['pad_right']:
            rpad_sym = self.ngrams_kwargs.get("right_pad_symbol")
            self.vocabulary[rpad_sym] = self.vocabulary.cutoff

    def _enumerate_ngram_orders(self):
        return enumerate(range(self.order, 1, -1))

    def train_counts(self, training_text):
        # Note here "1" indicates an empty vocabulary!
        # See NgramModelVocabulary __len__ method for more.
        if len(self.vocabulary) <= 1:
            raise EmptyVocabularyError("Cannot start counting ngrams until "
                                       "vocabulary contains more than one item.")

        for sent in training_text:
            checked_sent = (self.check_against_vocab(word) for word in sent)
            sent_start = True
            for ngram in self.to_ngrams(checked_sent):
                context, word = tuple(ngram[:-1]), ngram[-1]

                if sent_start:
                    for context_word in context:
                        self.unigrams[context_word] += 1
                    sent_start = False

                for trunc_index, ngram_order in self._enumerate_ngram_orders():
                    trunc_context = context[trunc_index:]
                    # note that above line doesn't affect context on first iteration
                    self.ngrams[ngram_order][trunc_context][word] += 1
                self.unigrams[word] += 1

    def check_against_vocab(self, word):
        if word in self.vocabulary:
            return word
        return self.unk_label

    def to_ngrams(self, sequence):
        """Wrapper around util.ngrams with usefull options saved during initialization.
        :param sequence: same as nltk.util.ngrams
        :type sequence: any iterable
        """
        return ngrams(sequence, self.order, **self.ngrams_kwargs)
    
NEG_INF = float("-inf")


class BaseNgramModel(object):
    """An example of how to consume NgramCounter to create a language model.
    This class isn't intended to be used directly, folks should inherit from it
    when writing their own ngram models.
    """

    def __init__(self, ngram_counter):

        self.ngram_counter = ngram_counter
        # for convenient access save top-most ngram order ConditionalFreqDist
        self.ngrams = ngram_counter.ngrams[ngram_counter.order]
        self._ngrams = ngram_counter.ngrams
        self._order = ngram_counter.order

        self._check_against_vocab = self.ngram_counter.check_against_vocab

    def check_context(self, context):
        """Makes sure context not longer than model's ngram order and is a tuple."""
        if len(context) >= self._order:
            raise ValueError("Context is too long for this ngram order: {0}".format(context))
        # ensures the context argument is a tuple
        return tuple(context)

    def score(self, word, context):
        """
        This is a dummy implementation. Child classes should define their own
        implementations.
        :param word: the word to get the probability of
        :type word: str
        :param context: the context the word is in
        :type context: Tuple[str]
        """
        return 0.5

    def logscore(self, word, context):
        """
        Evaluate the log probability of this word in this context.
        This implementation actually works, child classes don't have to
        redefine it.
        :param word: the word to get the probability of
        :type word: str
        :param context: the context the word is in
        :type context: Tuple[str]
        """
        score = self.score(word, context)
        if score == 0.0:
            return NEG_INF
        return log(score, 2)

    def entropy(self, text):
        """
        Calculate the approximate cross-entropy of the n-gram model for a
        given evaluation text.
        This is the average log probability of each word in the text.
        :param text: words to use for evaluation
        :type text: Iterable[str]
        """

        normed_text = (self._check_against_vocab(word) for word in text)
        H = 0.0     # entropy is conventionally denoted by "H"
        processed_ngrams = 0
        for ngram in self.ngram_counter.to_ngrams(normed_text):
            context, word = tuple(ngram[:-1]), ngram[-1]
            H += self.logscore(word, context)
            processed_ngrams += 1
        return - (H / processed_ngrams)

    def perplexity(self, text):
        """
        Calculates the perplexity of the given text.
        This is simply 2 ** cross-entropy for the text.
        :param text: words to calculate perplexity of
        :type text: Iterable[str]
        """

        return pow(2.0, self.entropy(text))


class MLENgramModel(BaseNgramModel):
    """Class for providing MLE ngram model scores.
    Inherits initialization from BaseNgramModel.
    """

    def score(self, word, context):
        """Returns the MLE score for a word given a context.
        Args:
        - word is expcected to be a string
        - context is expected to be something reasonably convertible to a tuple
        """
        context = self.check_context(context)
        return self.ngrams[context].freq(word)

class LidstoneNgramModel(BaseNgramModel):
    """Provides Lidstone-smoothed scores.
    In addition to initialization arguments from BaseNgramModel also requires
    a number by which to increase the counts, gamma.
    """

    def __init__(self, gamma, *args):
        super(LidstoneNgramModel, self).__init__(*args)
        self.gamma = gamma
        # This gets added to the denominator to normalize the effect of gamma
        self.gamma_norm = len(self.ngram_counter.vocabulary) * gamma

    def score(self, word, context):
        context = self.check_context(context)
        context_freqdist = self.ngrams[context]
        word_count = context_freqdist[word]
        ctx_count = context_freqdist.N()
        return (word_count + self.gamma) / (ctx_count + self.gamma_norm)

class LaplaceNgramModel(LidstoneNgramModel):
    """Implements Laplace (add one) smoothing.
    Initialization identical to BaseNgramModel because gamma is always 1.
    """

    def __init__(self, *args):
        super(LaplaceNgramModel, self).__init__(1, *args)
    
if __name__ == '__main__':

    with open('../data/ngrams.txt') as f:
        ngrams_txt = json.load(f)
        
    corpora = list()
    words = list()
    for key, value in ngrams_txt.items():
        ws = key.split()
        corpora.append(ws)
        words.extend(ws)

    print("Collected %d ngrams with %d words" % (len(corpora), len(words)))
    #with open("../out/corpora.pkl", "wb") as f:
    #    pickle.dump(corpora, f)

    vocab = build_vocabulary(1, words)
    print("Vocabulary built")
    counter = count_ngrams(5, vocab, corpora)
    print("Counter ready")
    with open(config.ngram_model_path, "wb") as f:
        pickle.dump(counter, f)
