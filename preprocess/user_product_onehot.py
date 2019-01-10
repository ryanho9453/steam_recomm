import os
import sys
import numpy as np
import pandas as pd
import json
from scipy.sparse import csc_matrix
from random import shuffle
import random

script_path = os.path.dirname(os.path.abspath(__file__)) + '/'
package_path = script_path.replace('preprocess/', '')
data_path = package_path + 'data/'

sys.path.append(script_path)


def sample_item_from_raw():
    col_name = ['user', 'game', 'action', 'act_amount', 'unk']
    steam = pd.read_csv(data_path + 'raw/steam.csv', names=col_name)

    del steam['act_amount']
    del steam['unk']

    c = steam['game'].value_counts().iloc[:2000]
    d = c.reset_index()['index'].values.tolist()
    print(d)

    with open(data_path + 'top2000item.json', 'r') as f:
        top2000item = json.load(f)

    steam = steam[steam.action == 'purchase']
    sample_df = steam[steam['game'].isin(top2000item)]

    sample_df.to_csv(data_path + 'raw/steam_sample.csv')


class UserProductOnehot:

    def __init__(self):
        """
        preprocessing for fm

        1. sample 2000 best seller games
        2. sample 1 positive per user as test set
        3. sample 2 size neg per user as train set
        4. a dict to store the test item per user
        """

        self.user_id_converter = None
        self.game_id_converter = None

        self.user2id = None
        self.game2id = None

        self.num_game = None
        self.total_n_user = None

        self.sample_n_user = 3000

        # dtype = list
        with open(data_path + 'top2000item.json', 'r') as f:
            self.top2000item = json.load(f)

    def create(self):
        """
        up_mat.shape = (num_user, num_game)

        total num of user : 12,393
                     game : 5,155

                     action : 200,000
                     purchase : 125,911

        sample

        1 choose 1000 user

        """
        def one_hot(num, id):
            one_hot = [0] * num
            one_hot[id] = 1
            return one_hot

        # dataframe convert user product to id
        print('create user product onehot')

        col_name = ['user', 'game', 'action']
        df = pd.read_csv(package_path + 'data/raw/steam_sample.csv', names=col_name)

        print('Num of Action : ' + str(df.shape[0]))

        self.user_id_converter = self.__gen_user_id_converter(df)
        self.game_id_converter = self.__gen_game_id_converter(df)

        user2item = dict()

        for userid in range(self.sample_n_user):
            user2item[str(userid)] = []

        test_set = user2item.copy()

        for idx, row in df.iterrows():
            userid = self.user2id[str(row['user'])]
            gameid = self.game2id[row['game']]

            if userid < self.sample_n_user:
                user2item[str(userid)].append(gameid)

        print('user2item done')

        with open(data_path + 'user2item.json', 'w') as f:
            json.dump(user2item, f)


        all_item = [i for i in range(self.num_game)]

        n_test_neg = 50

        output = []
        for userid in range(self.sample_n_user):
            print('UserID :' + str(userid))
            pos = user2item[str(userid)]

            n_pos = len(pos)

            neg = list(set(all_item) - set(pos))

            test_pos = pos.pop(random.randint(0, n_pos-1))

            shuffle(neg)
            train_neg = neg[:2 * n_pos]
            test_neg = neg[2 * n_pos: 2 * n_pos + n_test_neg]

            # test_set = {userid : [pos_item, neg_item, neg ......]
            test_set[str(userid)].append(test_pos)
            test_set[str(userid)] += test_neg

            for itemid in pos:
                record = one_hot(self.sample_n_user, userid) + one_hot(self.num_game, itemid) + [1]
                output.append(record)

            for neg in train_neg:
                record = one_hot(self.sample_n_user, userid) + one_hot(self.num_game, neg) + [-1]
                output.append(record)

        output = np.array(output, dtype=np.int8).reshape((-1, self.sample_n_user + self.num_game + 1))
        with open(data_path + 'test_set.json', 'w') as f:
            json.dump(test_set, f)

        np.save(package_path + 'data/user_item_sparse.npy', output)

        y = output[:, -1]
        x = np.delete(output, -1, axis=1)

        del output

        x_csc = csc_matrix(x)

        return x_csc, y, test_set

    def load(self):
        sparse = np.load(package_path + 'data/user_item_sparse.npy')

        with open(data_path + 'test_set.json', 'r') as f:
            test_set = json.load(f)

        y = sparse[:, -1]
        x = np.delete(sparse, -1, axis=1)

        x_csc = csc_matrix(x)

        return x_csc, y, test_set

    def __gen_user_id_converter(self, df):
        user = df['user'].unique().tolist()

        self.total_n_user = len(user)

        print('Num of Users : %s' % self.total_n_user)

        user2id = dict()
        id = 0
        for i in user:
            user2id[str(i)] = id
            id += 1

        self.user2id = user2id

        id2user = {str(id): user for user, id in user2id.items()}

        word_id_converter = {'user2id': user2id, 'id2user': id2user}

        with open(data_path + 'word_id_converter.json', 'w') as f:
            json.dump(word_id_converter, f)

        return word_id_converter

    def __gen_game_id_converter(self, df):
        game = df['game'].unique().tolist()

        self.num_game = len(game)

        print('Num of Games : %s' % self.num_game)

        game2id = dict()
        id = 0
        for i in game:
            game2id[i] = id
            id += 1

        self.game2id = game2id

        id2game = {str(id): game for game, id in game2id.items()}

        game_id_converter = {'game2id': game2id, 'id2game': id2game}

        with open(data_path + 'game_id_converter.json', 'w') as f:
            json.dump(game_id_converter, f)

        return game_id_converter

