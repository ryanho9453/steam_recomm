import json
import sys
import os
import argparse
# from models.LDA.lda_runner import lda_train

package_path = os.path.dirname(os.path.abspath(__file__)) + '/'

sys.path.append(package_path + 'models/LDA/')

from lda_runner import lda_train


if __name__ == "__main__":
    with open(package_path + 'config.json', 'r', encoding='utf8') as f:
        config = json.load(f)

    lda_train(config)
