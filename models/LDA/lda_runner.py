import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt

script_path = os.path.dirname(os.path.abspath(__file__)) + '/'
package_path = script_path.replace('models/LDA/', '')
preprocess_path = package_path + 'preprocess/'

sys.path.append(script_path)
sys.path.append(preprocess_path)

from lda_train import TrainSampler
from lda_predict import PredictSampler
from user_product_matrix import UserProductMatrix


"""
todo : coverage, diversity   

"""


def show_example(hit_example, miss_example):
    def list2string(wordlist):
        if len(wordlist) > 5:
            wordlist = wordlist[:5]

        output = '['
        for word in wordlist:
            output += word
            output += ', '
        output += ']'

        return output

    print('--- show recommendation example ---')
    print('--- [Hit]')
    for userid in hit_example.keys():
        answer = hit_example[str(userid)]['answer']
        recomm = list2string(hit_example[str(userid)]['recomm'])
        al_buy = list2string(hit_example[str(userid)]['already_buy'])

        print('<UserID>: %s , <Answer>: %s , <Recommend>: %s  , <Already_Buy>: %s' % (userid, answer, recomm, al_buy))

    print('--- [Miss]')
    for userid in miss_example.keys():
        answer = miss_example[str(userid)]['answer']
        recomm = list2string(miss_example[str(userid)]['recomm'])
        al_buy = list2string(miss_example[str(userid)]['already_buy'])

        print('<UserID>: %s , <Answer>: %s , <Recommend>: %s  , <Already_Buy>: %s' % (userid, answer, recomm, al_buy))


def show_performance(perform_record):
    name = []
    for metric in perform_record.keys():
        plt.plot(perform_record[metric])
        name.append(metric)

    plt.legend(name)
    plt.show()


def lda_train(config):
    n_topics = config['lda_model']['n_topics']
    alpha = config['lda_model']['alpha']
    beta = config['lda_model']['beta']
    maxiter = config['lda_model']['maxiter']

    eval_maxiter = config['lda_model']['eval']['maxiter']

    matrix = UserProductMatrix()
    train_mat, test_mat = matrix.create()

    # initialize
    print('initialize LDA model')
    maxlike = -1 * 10 ** 100

    perform_record = {'recall': [],
                      'hit_rate': []
                    }

    train_sampler = TrainSampler(n_topics=n_topics, alpha=alpha, beta=beta)

    for i, phi in enumerate(train_sampler.run(matrix=train_mat, maxiter=maxiter)):
        like = train_sampler.loglikelihood()

        predict_sampler = PredictSampler(phi, n_topics)
        predict_sampler.run(test_mat, maxiter=eval_maxiter)
        test_prec, test_recall, test_hr = predict_sampler.eval(test_mat)
        hit_example = predict_sampler.hit_example
        miss_example = predict_sampler.miss_example

        print('[Iter %s][Hit Rate] %s' % (i, test_hr))

        perform_record['recall'].append(test_recall)
        perform_record['hit_rate'].append(test_hr)

        # update best maximum likelihood and optimal phi = p(w| z)
        if like > maxlike:
            print('update')
            maxlike = like
            opt_phi = phi

    show_example(hit_example, miss_example)

    print('save lda model')
    np.save(script_path + 'saved_model/opt_phi.npy', opt_phi)

    show_performance(perform_record)
