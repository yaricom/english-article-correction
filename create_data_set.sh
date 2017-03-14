#!/bin/sh
#
# The script to create binary data set from data corpora
#

cd src

# Create train data set files
/usr/bin/env python3 data_set.py train --f_type tags

# Create validate data set files
/usr/bin/env python3 data_set.py validate --f_type tags

# Create test data set files
/usr/bin/env python3 data_set.py test --f_type tags
