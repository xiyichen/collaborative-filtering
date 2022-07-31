from args import parse_config
from main import get_trainer
from utils import *
import os
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
# from xgboost import XGBRegressor

def get_blender_model(**args):
    '''
    Wrapper for blender model. One of ['lr', 'xgboost', 'gbdt'].
    '''
    blender_model_type = args.get('blender_model_type')
    if blender_model_type == 'lr':
        return LinearRegression()
    elif blender_model_type == 'xgboost':
        return XGBRegressor()
    elif blender_model_type == 'gbdt':
        return GradientBoostingRegressor()

def predict_blended_values(X, blending_models, **args):
    '''
    Make predictions as an average of all k blending models on X.
    '''
    preds = []
    for i in range(args.get('k_fold')):
        preds.append(blending_models[i].predict(X))
    preds = np.array(preds).mean(axis=0).clip(args.get('min_rating'), args.get('max_rating'))
    return preds

def blend_for_submission(blending_models, **args):
    '''
    Make blended predictions for the test set for submission.
    '''
    print('Blending all predictions for submission')
    all_predictions = {}
    df_test = pd.read_csv(args.get('test_csv_path'))
    users_test, movies_test, ratings_test = extract_user_movie_rating_arrays(df_test)
    for model_type, final_pred_name in zip(args.get('model_types_blending'), args.get('final_pred_names')):
        pred = load_prediction(args.get('pred_folder'), final_pred_name)
        if pred.shape == (args.get('num_users'), args.get('num_movies')):
            pred = extract_prediction_from_full_matrix(pred, users_test, movies_test)
        all_predictions[model_type] = [pred]
    X, _ = get_X_fold(users_test, movies_test, ratings_test, np.arange(len(users_test)), all_predictions, 0, **args)
    blended = predict_blended_values(X, blending_models, **args)
    np.savetxt(os.path.join(args.get('pred_folder'), 'blended_test.txt'), blended)

def get_X_fold(users, movies, ratings, test_index, all_predictions, fold, **args):
    '''
    Get features of shape (n, m) for each blending fold. n denotes the number of samples in the test set,
    m denotes the number of models to be blended.
    '''
    users_test = users[test_index]
    movies_test = movies[test_index]
    ratings_test = ratings[test_index]

    # Load all predictions for the fold. Extract the test indices if the prediction were a full matrix.
    X_fold = []
    for model_type in args.get('model_types_blending'):
        prediction_fold = all_predictions[model_type][fold]
        if prediction_fold.shape == (args.get('num_users'), args.get('num_movies')):
            X_fold.append(extract_prediction_from_full_matrix(prediction_fold, users_test, movies_test))
        else:
            X_fold.append(prediction_fold)
    X_fold = np.array(X_fold).T
    y_fold = ratings_test
    return X_fold, y_fold

def blending(**args):
    df_all = pd.read_csv(args.get('train_csv_path'))
    # Extract user, movie, rating vectors from the dataframe.
    users, movies, ratings = extract_user_movie_rating_arrays(df_all)
    # Load predictions for all folds for all models
    all_predictions = load_all_predictions(**args)
    model_types_blending = args.get('model_types_blending')
    blending_models = []
    kfold = args.get('k_fold')

    kf = KFold(n_splits=kfold, shuffle=True, random_state=args.get('random_seed'))

    # Train blender models
    for idx, (_, test_index) in enumerate(kf.split(df_all)):
        X_fold, y_fold = get_X_fold(users, movies, ratings, test_index, all_predictions, idx, **args)
        
        print('Fold {}: Feature shape to be blended: {}'.format(idx+1, X_fold.shape))
        
        # Train a regressor to blend the current fold
        blender = get_blender_model(**args)
        blender.fit(X_fold, y_fold)
        blending_models.append(blender)

    # Test blender models
    rmses_all = 0
    rmse_all_single_models = {}
    for model_type in model_types_blending:
        rmse_all_single_models[model_type] = 0
    for idx, (_, test_index) in enumerate(kf.split(df_all)):
        X_fold, y_fold = get_X_fold(users, movies, ratings, test_index, all_predictions, idx, **args)

        preds = predict_blended_values(X_fold, blending_models, **args)
        rmse_fold = get_score(preds, y_fold)
        for i in range(len(model_types_blending)):
            rmse_all_single_models[model_types_blending[i]] += get_score(X_fold[:, i], y_fold)
        print('Blending RMSE for fold {} is {}'.format(idx+1, rmse_fold))
        rmses_all += rmse_fold
    print('Mean blending RMSE: {}'.format(rmses_all / kfold))
    for i in range(len(model_types_blending)):
        print('Mean RMSE for {} single model: {}'.format(model_types_blending[i], rmse_all_single_models[model_types_blending[i]] / kfold))

    # Blend predictions on the test set for submission
    if args.get('blend_for_submission'):
        blend_for_submission(blending_models, **args)

if __name__ == '__main__':
    args = parse_config()
    blending(**args)