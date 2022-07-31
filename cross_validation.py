from args import parse_config
from main import get_trainer
from utils import *
from sklearn.model_selection import KFold
import os

def cross_validation(**args):
    df_all = pd.read_csv(args.get('train_csv_path'))
    # Extract user, movie, rating vectors from the dataframe.
    users, movies, ratings = extract_user_movie_rating_arrays(df_all)
    num_users = args.get('num_users')
    num_movies = args.get('num_movies')
    device = args.get('device')
    cv_folder = args.get('cv_folder')
    base_model_name = args.get('model_name')
    if not os.path.exists(cv_folder):
        os.makedirs(cv_folder)

    kf = KFold(n_splits=args.get('k_fold'), shuffle=True, random_state=args.get('random_seed'))

    args['pred_folder'] = cv_folder
    args['ckpt_folder'] = cv_folder
    for idx, (train_index, test_index) in enumerate(kf.split(df_all)):
        users_train = users[train_index]
        movies_train = movies[train_index]
        ratings_train = ratings[train_index]

        users_test = users[test_index]
        movies_test = movies[test_index]
        ratings_test = ratings[test_index]

        print('Fold {}: Number of training sumples: {}, number of test samples: {}.'.format(idx+1, len(users_train), len(users_test)))

        args['model_name'] = base_model_name + '_fold_' + str(idx+1)
        trainer = get_trainer(**args)
        trainer.train(users_train, movies_train, ratings_train, users_test, movies_test, ratings_test, **args)

if __name__ == '__main__':
    args = parse_config()
    cross_validation(**args)