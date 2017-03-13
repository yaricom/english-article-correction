#!/bin/sh
#
# The script to generate final prediction results
#

cd src

# Do prediction with RandomForest
/usr/bin/env python3 predictor.py RandomForest --save_model --validate_model --save_labels

# Generate and save submission results
/usr/bin/env python3 results_generator.py
