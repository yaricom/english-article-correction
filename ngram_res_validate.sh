#!/bin/sh
#
# Builds and validates prediction results 
#

cd src

# Build predictions list
RESULTS_FILE="../out/submission_validate_ngram.txt"
TEXT_FILE="../data/sentence_test.txt"
/usr/bin/env python3 ngram_res.py --out_file $RESULTS_FILE \
                                  --test_sentences_file $TEXT_FILE \
                                  --model_file ../out/counter.pkl

# Evaluate results
/usr/bin/env python3 evaluate.py --results_file $RESULTS_FILE \
                                --test_sentences_file $TEXT_FILE \
                                --test_corrections_file ../data/corrections_test.txt
