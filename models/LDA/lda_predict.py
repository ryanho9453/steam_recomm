
import sys
import os
import numpy as np
from scipy.special import gammaln
import json
import random
import matplotlib.pyplot as plt

script_path = os.path.dirname(os.path.abspath(__file__)) + '/'
package_path = script_path.replace('models/LDA/', '')
preprocess_path = package_path + 'preprocess/'
data_path = package_path + 'data/'
utils_path = package_path + 'utils/'

sys.path.append(script_path)
sys.path.append(preprocess_path)
sys.path.append(utils_path)


def get_top_n_col_id(mat_2d, top_n):
    """
    find the col_idx of top n value in matrix
    """

    max_value_in_col = np.amax(mat_2d, axis=0)

    top_n_col_idx = np.argpartition(max_value_in_col, -1 * top_n)[-1 * top_n:][::-1].tolist()

    return top_n_col_idx


def sample_index(p):
    """
    Sample from the Multinomial distribution and return the sample index.
    """

    A = np.random.multinomial(1, p).argmax()

    return A


def item_indices(vec):
    """
    Turn a user vector of size vocab_size to a sequence
    of item indices. The item indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.

             w1      w2      w3
    doc1 [   0       6        3

    return [w2*6  , w3*3 .....

    """

    # 若doc中有 a個Wa b個Wb ,則傳回Wa a次, 傳回Wb b次
    # idx 是td_matrix中 word的index, 只是這只取doc 中 nonzero 的字 )
    for idx in vec.nonzero()[0]:        # 取出doc 包含的word的id
        for i in range(int(vec[idx])):  # 若doc 中有n個wi, 傳回n 個 wi
            yield idx


def log_multi_beta(alpha, K=None):
    """
    Logarithm of the multinomial beta function.
    """
    if K is None:
        # alpha is assumed to be a vector
        return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    else:
        # alpha is assumed to be a scalar
        return K * gammaln(alpha) - gammaln(K*alpha)


class PredictSampler:
    """
    input matrix = [[1, 1, 0, 1],   (user1)
                   [1, 1, 0, 0],    (user2)
                   [1, 0, 1, 0]]    (user3)


    for user1 , we take itemid [0, 1] to do gibbs sampling,
    and take "3" to be the positive sample
    after gibbs sampling , we ask for 2 recommendation
    and the model have to recommend 2 items beside [0, 1]
    and hit the positive sample "3" to count one hit


    """
    def __init__(self, phi, n_topics, alpha=0.1):

        self.ndz = None
        self.nz = None

        # phi.shape = (z, w)
        self.phi = phi

        self.alpha = alpha

        self.n_topics = n_topics

        self.n_user = None
        self.vocab_size = None

        self.n_predict_for_eval = 10
        self.n_sample_for_eval = 1

        self.pos_sample = None

        with open(data_path + 'game_id_converter.json', 'r') as f:
            game_id_converter = json.load(f)

        self.id2game = game_id_converter['id2game']

        self.hit_example = dict()
        self.miss_example = dict()

    def run(self, matrix, maxiter):
        """
        do the gibbs sampling except of the positive example


        """
        self.n_user, self.vocab_size = matrix.shape

        # get one positive sample from each user for evaluation
        self._gen_pos_sample(matrix)

        self._initialize(matrix)

        self._check_if_zero(self.ndz, 'after init')

        for it in range(maxiter):
            for d in range(self.n_user):
                pos_sample = self.pos_sample[str(d)]
                for i, w in enumerate(item_indices(matrix[d, :])):
                    # do gibbs sampling for all items except the positive sample
                    if w not in pos_sample:
                        z = self.topics[(d, i)]
                        self.ndz[d, z] -= 1
                        self.nz[z] -= 1
                        self.nd[d] -= 1

                        p_z = self._conditional_distribution(d, w)
                        z = sample_index(p_z)

                        self.ndz[d, z] += 1
                        self.nz[z] += 1
                        self.nd[d] += 1
                        self.topics[(d, i)] = z

    def _check_if_zero(self, ndz, where):
        for d in range(self.n_user):
            p_zd = ndz[d, :][:, np.newaxis]
            if int(np.sum(p_zd, axis=0)[0]) == 0:
                print('zd empty in ' + str(where))

    def eval(self, matrix):
        n_tp = 0
        n_predict = 0
        n_pos = 0

        self._check_if_zero(self.ndz, 'eval start')

        for d in range(self.n_user):
            # type(pos_sample) = list
            pos_sample = self.pos_sample[str(d)]

            p_zd = self.ndz.copy()
            p_zd = p_zd[d, :][:, np.newaxis]
            p_zd /= int(np.sum(p_zd, axis=0)[0])

            score = self.phi * p_zd

            user_vec = matrix[d, :]
            itemid_list_true = np.nonzero(user_vec)[0].tolist()

            used_pos = list(set(itemid_list_true) - set(pos_sample))

            recomm = self._get_recommend(score, used_pos)

            true_pos = list(set(recomm) & set(pos_sample))

            if len(true_pos) == 1 and len(self.hit_example.keys()) <=5:
                self._collect_example(hit=True, userid=d,used_pos=used_pos, recomm=recomm, pos_sample=pos_sample[0])

            elif len(true_pos) == 0 and len(self.miss_example.keys()) <=5:
                self._collect_example(hit=False, userid=d, used_pos=used_pos, recomm=recomm, pos_sample=pos_sample[0])

            n_tp += len(true_pos)
            n_predict += self.n_predict_for_eval
            n_pos += self.n_sample_for_eval

        precision = n_tp / n_predict
        recall = n_tp / n_pos
        hit_rate = n_tp / self.n_user

        return precision, recall, hit_rate

    def _collect_example(self, hit, userid, used_pos, recomm, pos_sample):
        """
        collect example of hit and miss and translate the itemid back to item name
        """
        if hit is True:
            self.hit_example[str(userid)] = {'already_buy': [], 'recomm': []}
            for id in used_pos:
                self.hit_example[str(userid)]['already_buy'].append(self.id2game[str(id)])

            for id in recomm:
                self.hit_example[str(userid)]['recomm'].append(self.id2game[str(id)])

            self.hit_example[str(userid)]['answer'] = self.id2game[str(pos_sample)]

        elif hit is False:
            self.miss_example[str(userid)] = {'already_buy': [], 'recomm': []}
            for id in used_pos:
                self.miss_example[str(userid)]['already_buy'].append(self.id2game[str(id)])

            for id in recomm:
                self.miss_example[str(userid)]['recomm'].append(self.id2game[str(id)])

            self.miss_example[str(userid)]['answer'] = self.id2game[str(pos_sample)]

    def _gen_pos_sample(self, matrix):
        pos_sample = dict()
        for d in range(self.n_user):
            itemlist = np.nonzero(matrix[d, :])[0].tolist()
            random.shuffle(itemlist)
            sample = itemlist[:self.n_sample_for_eval]
            pos_sample[str(d)] = sample

        self.pos_sample = pos_sample

    def _get_recommend(self, score, used_pos):
        """
        recommend n_predict_for_eval of recommendation, excluding the positive used in gibbs sampling

        """

        # score.shape = (z, w)

        max_value_in_col = np.amax(score, axis=0)

        sorted_id = np.argsort(max_value_in_col).tolist()[::-1]

        recommend = []
        n_recomm = 0
        while n_recomm < self.n_predict_for_eval:
            id = sorted_id.pop(0)
            if id not in used_pos:
                recommend.append(id)
                n_recomm += 1

        # print(sorted_id)
        # recommend = list(set(sorted_id) - set(used_pos))[::-1][:self.n_predict_for_eval]

        return recommend

    def _initialize(self, matrix):

        # number of times document m and topic z co-occur
        self.ndz = np.zeros((self.n_user, self.n_topics))

        # n_words in docs
        self.nd = np.zeros(self.n_user)

        # n_words in topics
        self.nz = np.zeros(self.n_topics)
        self.topics = {}

        for d in range(self.n_user):
            pos_sample = self.pos_sample[str(d)]
            for i, w in enumerate(item_indices(matrix[d, :])):
                # choose an arbitrary topic as first topic for word i
                if w not in pos_sample:
                    z = np.random.randint(self.n_topics)
                    self.ndz[d, z] += 1
                    self.nd[d] += 1
                    self.nz[z] += 1
                    self.topics[(d, i)] = z

    def _conditional_distribution(self, d, w):
        """
        Conditional distribution (vector of size n_topics).

        """
        # phi.shape = (z, w)
        left = self.phi[:, w]

        right = (self.ndz[d, :] + self.alpha) / \
                (self.nd[d] + self.alpha * self.n_topics)
        p_z = left * right

        # normalize to obtain probabilities
        p_z /= np.sum(p_z)

        return p_z





