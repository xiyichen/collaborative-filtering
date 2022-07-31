from args import parse_config
from utils import *
import numpy as np
import os

def main(**args):
    '''
    Generate kaggle format for submission
    '''
    preds = np.loadtxt(os.path.join(args.get('pred_folder'), args.get('model_name') + '.txt'))
    df_test = pd.read_csv(args.get('test_csv_path'))
    users_test, movies_test, _ = extract_user_movie_rating_arrays(df_test)
    if args.get('save_pred_type') == 'full':
        preds = extract_prediction_from_full_matrix(preds, users_test, movies_test)
    
    with open('./submission.csv', 'w') as f:
        f.write('Id,Prediction\n')
        for (user, movie, pred) in zip(users_test, movies_test, preds):
            f.write("r{}_c{},{}\n".format(user+1, movie+1, pred))

if __name__ == '__main__':
    args = parse_config()
    main(**args)