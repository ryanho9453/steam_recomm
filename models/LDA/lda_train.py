"""
(C) Mathieu Blondel - 2010
License: BSD 3 clause

Implementation of the collapsed Gibbs sampler for
Latent Dirichlet Allocation, as described in

Finding scientifc topics (Griffiths and Steyvers)
"""

import sys
import os
import numpy as np
from scipy.special import gammaln
import json

script_path = os.path.dirname(os.path.abspath(__file__)) + '/'
package_path = script_path.replace('models/LDA/', '')
preprocess_path = package_path + 'preprocess/'
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


def word_indices(vec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
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


class TrainSampler:
    def __init__(self, n_topics, alpha=0.1, beta=0.1):
        """
        n_topics: desired number of topics
        alpha: a scalar (FIXME: accept vector of size n_topics)
        beta: a scalar (FIME: accept vector of size vocab_size)
        """
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta

    def _initialize(self, matrix):
        self.n_docs, self.vocab_size = matrix.shape

        # number of times document m and topic z co-occur
        self.ndz = np.zeros((self.n_docs, self.n_topics))
        # number of times topic z and word w co-occur
        self.nzw = np.zeros((self.n_topics, self.vocab_size))

        # n_words in docs
        self.nd = np.zeros(self.n_docs)

        # n_words in topics
        self.nz = np.zeros(self.n_topics)
        self.topics = {}

        for d in range(self.n_docs):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, w in enumerate(word_indices(matrix[d, :])):
                # choose an arbitrary topic as first topic for word i
                z = np.random.randint(self.n_topics)
                self.ndz[d, z] += 1
                self.nd[d] += 1
                self.nzw[z, w] += 1
                self.nz[z] += 1
                self.topics[(d, i)] = z

    def _conditional_distribution(self, d, w):
        """
        Conditional distribution (vector of size n_topics).

        2 way to define the prior , in one cluster, not_in multiple clusters

        p_z_with_prior = [......., in=0.8 , ....]

        """

        # left = p( w | z )
        left = (self.nzw[:, w] + self.beta) / \
            (self.nz + self.beta * self.vocab_size)
        right = (self.ndz[d, :] + self.alpha) / \
            (self.nd[d] + self.alpha * self.n_topics)
        p_z = left * right

        # normalize to obtain probabilities
        p_z /= np.sum(p_z)

        return p_z

    def run(self, matrix, maxiter=30):
        """
        input_mat.shape = (#doc, #word)

        Run the Gibbs sampler.
        """
        n_docs, vocab_size = matrix.shape

        self._initialize(matrix)

        for it in range(maxiter):

            for d in range(n_docs):
                # 在doc m下的第i個字的wordid -- w
                for i, w in enumerate(word_indices(matrix[d, :])):
                    z = self.topics[(d, i)]
                    self.ndz[d, z] -= 1
                    self.nd[d] -= 1
                    self.nzw[z, w] -= 1
                    self.nz[z] -= 1

                    p_z = self._conditional_distribution(d, w)
                    z = sample_index(p_z)

                    self.ndz[d, z] += 1
                    self.nd[d] += 1
                    self.nzw[z, w] += 1
                    self.nz[z] += 1
                    self.topics[(d, i)] = z

            # FIXME: burn-in and lag!
            yield self.__gen_phi()

    def recall(self, matrix, phi, n_predict):
        """
        recall = TP / positive

        positve = purchase

        n_predict will effect accuracy

        """

        n_pos = 0
        n_tp = 0

        for d in range(self.n_docs):
            p_zd = self.ndz.copy()
            p_zd = p_zd[d, :][:, np.newaxis]
            p_zd /= int(np.sum(p_zd, axis=0)[0])

            score = phi * p_zd
            wordid_list_predict = get_top_n_col_id(score, n_predict)

            doc_vec = matrix[d, :]
            wordid_list_true = np.nonzero(doc_vec)[0].tolist()

            n_pos += len(wordid_list_true)

            true_pos = list(set(wordid_list_predict) & set(wordid_list_true))

            n_tp += len(true_pos)

        return n_tp / n_pos

    def precision(self, matrix, phi, n_predict):
        """
        precision = TP / predicted positive

        positve = purchase

        n_predict will effect accuracy

        """

        total_predict = 0
        total_tp = 0

        for d in range(self.n_docs):
            p_zd = self.ndz.copy()
            p_zd = p_zd[d, :][:, np.newaxis]
            p_zd /= int(np.sum(p_zd, axis=0)[0])

            score = phi * p_zd
            wordid_list_predict = get_top_n_col_id(score, n_predict)

            doc_vec = matrix[d, :]
            wordid_list_true = np.nonzero(doc_vec)[0].tolist()

            true_pos = list(set(wordid_list_predict) & set(wordid_list_true))

            total_tp += len(true_pos)
            total_predict += n_predict

        return total_tp / total_predict

    def loglikelihood(self):
        """
        Compute the likelihood that the model generated the data.
        """

        lik = 0

        for z in range(self.n_topics):
            lik += log_multi_beta(self.nzw[z, :]+self.beta)
            lik -= log_multi_beta(self.beta, self.vocab_size)

        for d in range(self.n_docs):
            lik += log_multi_beta(self.ndz[d, :]+self.alpha)
            lik -= log_multi_beta(self.alpha, self.n_topics)

        return lik

    def __gen_phi(self):
        """
        Compute phi = p(w|z).
                pzw = p(z|w)

        output.shape = (z, w)

        """

        phi = self.nzw + self.beta
        phi /= np.sum(phi, axis=1)[:, np.newaxis]

        return phi

    def __gen_pzw(self):
        """
        Compute pzw = p(z|w)

        output.shape = (w, z)

        """

        p_zw = self.nzw.copy()
        p_zw /= np.sum(p_zw, axis=0)[np.newaxis, :]

        return p_zw.T

    def __gen_theta(self):
        """
        Compute theta = p(z| d)

        output.shape = (w, z)

        """

        theta = self.ndz + self.alpha
        theta /= np.sum(theta, axis=1)[:, np.newaxis]

        return theta
