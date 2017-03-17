#!/bin/sh
#
# This is script to perform prediction over validation data set
# and to evaluate prediction results
#

cd src

# Do prediction over validation data set
/usr/bin/env python3 predictor.py RandomForest --test_data ../out/intermediate/validate_features.npy \
						--save_labels \
						--validate_model

# Generate results
RESULTS_FILE="../out/submission_validate_tags.txt"
TEXT_FILE="../data/sentence_test.txt"
/usr/bin/env python3 results_generator.py --out_file $RESULTS_FILE \
					--test_sentences_file $TEXT_FILE \
					--test_parse_tree_file ../data/parse_test.txt \
                     --f_type tags

# Evaluate results
/usr/bin/env python3 evaluate.py --results_file $RESULTS_FILE \
				--test_sentences_file $TEXT_FILE \
				--test_corrections_file ../data/corrections_test.txt
