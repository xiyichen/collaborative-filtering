from args import parse_config
from utils import *
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from VAE_trainer import VAE_trainer
from AE_trainer import AE_trainer
from BFM_trainer import BFM_trainer
from RBSVD_trainer import RBSVD_trainer
from IterativeSVD_trainer import IterativeSVD_trainer
from SVD_trainer import SVD_trainer
from ALS_trainer import ALS_trainer
from NCF_trainer import NCF_trainer
import os

def get_trainer(**args):
    '''
    Trainer function wrapper. One of ['ae', 'vae', 'bfm', 'iterative_svd', 'rbsvd', 'svd', 'als', 'ncf'].
    '''
    model_type = args.get('model_type')
    if model_type == 'ae':
        return AE_trainer(**args)
    elif model_type == 'vae':
        return VAE_trainer(**args)
    elif model_type == 'bfm':
        return BFM_trainer(**args)
    elif model_type == 'rbsvd':
        return RBSVD_trainer(**args)
    elif model_type == 'iterative_svd':
        return IterativeSVD_trainer(**args)
    elif model_type == 'svd':
        return SVD_trainer(**args)
    elif model_type == 'als':
        return ALS_trainer(**args)
    elif model_type == 'ncf':
        return NCF_trainer(**args)

def train(**args):
    df_all = pd.read_csv(args.get('train_csv_path'))
    final_model = args.get('final_model')
    if not os.path.exists(args.get('ckpt_folder')):
        os.makedirs(args.get('ckpt_folder'))
    if not os.path.exists(args.get('pred_folder')):
        os.makedirs(args.get('pred_folder'))

    if final_model:
        # Training final model using all available data, loading the public test set as df_test (without ground truth).
        df_train = df_all
        df_test = pd.read_csv(args.get('test_csv_path'))
    else:
        # Split the data into train and test set
        df_train, df_test = train_test_split(df_all, test_size=args.get('test_size'), random_state=args.get('random_seed'))
    # Extract user, movie, rating vectors from the dataframe.
    users_train, movies_train, ratings_train = extract_user_movie_rating_arrays(df_train)
    users_test, movies_test, ratings_test = extract_user_movie_rating_arrays(df_test)

    print('Number of training sumples: {}, number of test samples: {}.'.format(len(users_train), len(users_test)))

    trainer = get_trainer(**args)
    trainer.train(users_train, movies_train, ratings_train, users_test, movies_test, ratings_test, **args)

if __name__ == '__main__':
    args = parse_config()
    train(**args)