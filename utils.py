from torch import nn
from sklearn.metrics import mean_squared_error
import math
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def init_weights(layers, init_type):
    '''
    Initialize weights for a NN layer. if init_type is not None, it should be either 
    "xavier" (https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) or
    "kaiming" (https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_)
    '''
    for layer in layers:
        if init_type == 'xavier':
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(layer.weight)

        # Normal Initialization for Biases
        layer.bias.data.normal_(0.0, 0.001)

def get_activation(activation_type):
    '''
    Activation function wrapper. One of ('relu', 'leakyrelu', 'sigmoid', 'tanh')
    '''
    if activation_type == 'relu':
        return nn.ReLU()
    elif activation_type == 'leakyrelu':
        return nn.LeakyReLU()
    elif activation_type == 'sigmoid':
        return nn.Sigmoid()
    elif activation_type == 'tanh':
        return nn.Tanh()

def get_all_user_movie_indices():
    '''
    Get all index combinations of users in [0, num_users) and movies in [0, num_movies)
    '''
    full_matrix_users = []
    for i in range(self.num_users):
        for j in range(self.num_movies):
            full_matrix_users.append(i)
    full_matrix_movies = []
    for i in range(self.num_users):
        for j in range(self.num_movies):
            full_matrix_movies.append(j)
    return full_matrix_users, full_matrix_movies

def get_dataloader(users_train, movies_train, ratings_train, impute=0, **args):
    '''
    Convert available rating records to a num_users*num_movies matrix, return its pytorch dataloder, 
    data torch tensor, mask tensor, and user id tensor
    '''
    num_users = args.get('num_users')
    num_movies = args.get('num_movies')
    device = args.get('device')
    data_zeros = np.full((num_users, num_movies), impute)
    data_mask = np.full((num_users, num_movies), impute)

    for i, (user, movie) in enumerate(zip(users_train, movies_train)):
        data_zeros[user][movie] = ratings_train[i]
        data_mask[user][movie] = 1

    data_torch = torch.tensor(data_zeros, device=device).float()
    mask_torch = torch.tensor(data_mask, device=device)
    user_id_torch = torch.tensor(list(range(0, num_users)), device=device)
    dataloader = DataLoader(TensorDataset(data_torch, mask_torch, user_id_torch), batch_size=args.get('batch_size'), shuffle=True)
    return dataloader, data_torch, mask_torch, user_id_torch

rmse = lambda x, y: math.sqrt(mean_squared_error(x, y))
def get_score(predictions, target_values):
    '''
    Get RMSE score between predictions and ground truth
    '''
    return rmse(predictions, target_values)

def extract_prediction_from_full_matrix(reconstructed_matrix, users, movies):
    '''
    returns predictions for the users-movies combinations specified based on a full m*n matrix
    '''
    assert(len(users) == len(movies)), "users-movies combinations specified should have equal length"
    predictions = np.zeros(len(users))

    for i, (user, movie) in enumerate(zip(users, movies)):
        predictions[i] = reconstructed_matrix[user][movie]

    return predictions

def extract_user_movie_rating_arrays(df):
    '''
    Convert dataframe in the format of 'rx_cy' rows into lists of users, movies, ratings
    '''
    users = list()
    movies = list()
    for user, movie in df.Id.str.extract('r(\d+)_c(\d+)').values:
        users.append(int(user) - 1)
        movies.append(int(movie) - 1)
    users = np.array(users)
    movies = np.array(movies)
    ratings = df['Prediction'].values
    return users, movies, ratings

def stand_norm(A):
    """
    Normalize a matrix such that it has mean 0 and variance 1, columnwise
    """
    col_mean = np.nanmean(A, axis=0)
    col_std = np.nanstd(A, axis=0)
    A_n = (A-col_mean)/col_std
    return A_n, col_mean, col_std

def de_norm(A, col_mean, col_std):
    """
    Denormalize a matrix
    """
    return (A*col_std)+col_mean

def impute_matrix(A, impute_method="zero"):
    """
    Impute missing values of a matrix.
    impute_method is either "zero" for zero imputation or "col_mean" for columnwise mean imputation
    """
    if impute_method == "zero":
        A_i = np.nan_to_num(A, nan=0.0)
        return A_i
    else:
        A_i = np.copy(A)
        item_mean = np.nanmean(A_i, axis=0)
        na_idx = np.where(np.isnan(A_i))
        A_i[na_idx] = np.take(item_mean, na_idx[1])
        return A_i