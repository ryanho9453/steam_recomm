import os
import sys
import numpy as np
import pandas as pd
import json

script_path = os.path.dirname(os.path.abspath(__file__)) + '/'
package_path = script_path.replace('preprocess/', '')
data_path = package_path + 'data/'

sys.path.append(script_path)


class UserProductMatrix:
    """
    preprocessing for LDA
    """

    def __init__(self):

        self.user_id_converter = None
        self.game_id_converter = None

        self.user2id = None
        self.game2id = None

        self.num_game = None
        self.num_user = None

        self.up_mat = None

    def create(self):
        """
        up_mat.shape = (num_user, num_game)

        total num of user : 12,393
                  of game : 5155

        after filter --- user : 6693


        """
        train_size = 5019

        print('create user product matrix')

        col_name = ['user', 'game', 'action', 'act_amount', 'unk']
        df = pd.read_csv(package_path + 'data/raw/steam.csv', names=col_name)

        del df['unk']

        self.user_id_converter = self.__gen_user_id_converter(df)
        self.game_id_converter = self.__gen_game_id_converter(df)

        up_mat = np.zeros((self.num_user, self.num_game), dtype=np.int8)

        for index, row in df.iterrows():
            if row['action'] == 'purchase':
                row_idx = self.user2id[str(row['user'])]
                col_idx = self.game2id[row['game']]

                up_mat[row_idx, col_idx] = 1

        print('save user product matrix')

        np.random.shuffle(up_mat)

        print('before filter')
        print(up_mat.shape)

        up_mat = self.__filter_user(up_mat)

        print('after filter')
        print(up_mat.shape)

        np.save(data_path + 'user_product_matrix.npy', up_mat)

        train_mat = up_mat[0:train_size, :]
        test_mat = up_mat[train_size:, :]

        return train_mat, test_mat

    def __filter_user(self, up_mat):
        total_game = np.sum(up_mat, axis=1)
        del_list = np.where(total_game < 2)[0].tolist()
        output = np.delete(up_mat, del_list, axis=0)

        return output

    def load(self):
        up_mat = np.load(data_path + 'user_product_matrix.npy')

        print('save user product matrix')

        np.random.shuffle(up_mat)

        train_size = 7435

        train_mat = up_mat[0:train_size - 1, :]
        test_mat = up_mat[train_size:, :]

        return train_mat, test_mat

    def __gen_user_id_converter(self, df):
        user = df['user'].unique().tolist()

        self.num_user = len(user)

        # print('Num of Users : %s' % self.num_user)

        user2id = dict()
        id = 0
        for i in user:
            user2id[str(i)] = id
            id += 1

        self.user2id = user2id

        id2user = {str(id): user for user, id in user2id.items()}

        user_id_converter = {'user2id': user2id, 'id2user': id2user}

        with open(data_path + 'user_id_converter.json', 'w') as f:
            json.dump(user_id_converter, f)

        return user_id_converter

    def __gen_game_id_converter(self, df):
        game = df['game'].unique().tolist()

        self.num_game = len(game)

        # print('Num of Games : %s' % self.num_game)

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

