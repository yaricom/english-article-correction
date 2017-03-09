import json
from random import seed, random

with open('data/sentence_test.txt') as f:
    test_data = json.load(f)

seed(174567)

with open('submission_test.txt', 'wb') as f:
    json.dump([[('the', random()) if w in ['a', 'an'] else None for w in s] for s in test_data], f)
