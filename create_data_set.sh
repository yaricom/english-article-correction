#!/bin/sh
#
# The script to create binary data set from data corpora
#

cd src

# Create train data set files
/usr/bin/env python3 data_set.py train

# Create validate data set files
/usr/bin/env python3 data_set.py validate

# Create test data set files
/usr/bin/env python3 data_set.py test
