#!/bin/sh
#
# The script to generate final prediction results
#

cd src

# Create data set files
/usr/bin/env python3 data_set.py train --f_type tags
/usr/bin/env python3 data_set.py validate --f_type tags
/usr/bin/env python3 data_set.py test --f_type tags

# Do prediction with RandomForest
/usr/bin/env python3 predictor.py RandomForest --save_model --validate_model --save_labels

# Generate and save submission results
/usr/bin/env python3 results_generator.py --f_type tags
