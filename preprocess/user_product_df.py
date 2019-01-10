import os
import sys
import numpy as np
import pandas as pd
import json

script_path = os.path.dirname(os.path.abspath(__file__)) + '/'
package_path = script_path.replace('preprocess/', '')
data_path = package_path + 'data/'

sys.path.append(script_path)


class UserProductDF:
    """
    preprocessing for surprise

    """

    def __init__(self):

        self.user_id_converter = None
        self.game_id_converter = None

        self.user2id = None
        self.game2id = None

        self.num_game = None
        self.num_user = None

    def create(self):
        """
        up_mat.shape = (num_user, num_game)

        total num of user : 12,393
                  of game : 5155


        """

        print('create user product dataframe')

        col_name = ['user', 'game', 'action', 'act_amount', 'unk']
        raw_df = pd.read_csv(package_path + 'data/raw/steam.csv', names=col_name)

        del raw_df['unk']
        del raw_df['act_amount']

        self.user_id_converter = self.__gen_user_id_converter(raw_df)
        self.game_id_converter = self.__gen_game_id_converter(raw_df)

        # create new df filled with 0
        userid_list = self.user2id.keys()
        gameid_list = self.game2id.keys()

        filled_df = pd.DataFrame(columns=['user', 'game', 'action'])
        for userid in userid_list:
            for gameid in gameid_list:
                filled_df.append({'user': userid, 'game': gameid, 'action': 0}, ignore_index=True)

        # convert purchase in raw_df into filled_df
        raw_df = raw_df[raw_df.action == 'purchase']

        for index, row in raw_df.iterrows():
            userid = self.user2id[str(row['user'])]
            gameid = self.game2id[row['game']]
            filled_df.loc[(filled_df.user == userid) & (filled_df.game == gameid), ['action']] = 1

        print('save user product dataframe')
        filled_df.to_csv(package_path + 'data/user_product_df.csv')

        return filled_df

    def load(self):
        col_name = ['user', 'game', 'action']
        df = pd.read_csv(package_path + 'data/user_product_df.csv', names=col_name)

        return df

    def __gen_user_id_converter(self, df):
        user = df['user'].unique().tolist()

        self.num_user = len(user)

        print('Num of Users : %s' % self.num_user)

        user2id = dict()
        id = 0
        for i in user:
            user2id[str(i)] = id
            id += 1

        self.user2id = user2id

        id2user = {str(id): user for user, id in user2id.items()}

        return {'user2id': user2id, 'id2user': id2user}

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

        return {'game2id': game2id, 'id2game': id2game}
