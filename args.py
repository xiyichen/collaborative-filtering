import argparse
import torch

def parse_config(argv=None):
    '''
    Arguments list for training, cross validation, and blending for all of our models.
    '''
    parser = argparse.ArgumentParser(description='Parameters for collaborative filtering.')
    parser.add_argument('--final_model', action='store_true', help='Whether to train the model using the entire training set (for final submission)')
    parser.add_argument('--save_model', action='store_true', help='Whether to save trained model')
    parser.add_argument('--train_csv_path', default='./data_train.csv', type=str,
                        help='File path for the entire training set')
    parser.add_argument('--test_csv_path', default='./sampleSubmission.csv', type=str,
                        help='File path for the test set (no ground truth)')
    parser.add_argument('--random_seed', default=42, type=int,
                        help='Random seed for train/test split and k-fold cross validation. Using the same random seed could reproduce our experiment results')
    parser.add_argument('--num_users', default=10000, type=int, help='Number of users in the dataset')
    parser.add_argument('--num_movies', default=1000, type=int, help='Number of movies in the dataset')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), type=torch.device, help='device to train the neural networks')
    parser.add_argument('--ckpt_folder', default='checkpoints', type=str, help='Folder to store model files')
    parser.add_argument('--pred_folder', default='predictions', type=str, help='Folder to store predicted matrices')
    parser.add_argument('--save_pred_type', default=None, choices=['full', 'test_indices', None], help='If None, save no predictions. Otherwise, save either full reconstructed matrix or only ratings on indices ' + 
        'in the test set. Note that predicting the full reconstruction could be slow for some methods, e.g., it takes about 20 minutes for reconstructing all 10000x1000 entries with BFM_base+implicit.')
    parser.add_argument('--model_name', default='final_model', type=str, help='Name of the current model')
    parser.add_argument('--ckpt_path', default=None, type=str, help='Path of pre-trained model (for resume training or testing)')
    parser.add_argument('--beta_annealing_schedule', default='cyclic', choices=['linear', 'cyclic', None], type=str, 
        help='Choice of beta annealing schedule (or None to use a constant beta value, which is defined as beta_max)')
    parser.add_argument('--use_user_bias', action='store_true', help='Whether to use user bias')
    parser.add_argument('--use_movie_bias', action='store_true', help='Whether to use movie bias')
    parser.add_argument('--hidden_dimension', default=256, type=int, help='Number of neurons in the hidden layer for AE and VAE')
    parser.add_argument('--latent_dimension', default=32, type=int, help='Dimension of the latent space Z for AE and VAE')
    parser.add_argument('--dropout_p', default=0.5, type=float, help='Dropout probability for neural network models')
    parser.add_argument('--weight_init_type', default='xavier', choices=['xavier', 'kaiming', None], type=str, 
        help='Method to initialize layer weights for neural network models')
    parser.add_argument('--activation_type', default='tanh', choices=['tanh', 'sigmoid', 'leakyrelu', 'relu'], type=str, 
        help='Choice of activation for neural network models')
    parser.add_argument('--batch_size', default=1024, type=int, help='Batch size for neural network models')
    parser.add_argument('--num_iterations', default=2000, type=int, help='Number of iterations/epochs to train the model')
    parser.add_argument('--last_iteration', default=-1, type=int, help='Last trained epoch, for resume training of a neural network model')
    parser.add_argument('--init_lr', default=0.025, type=float, help='Initial learning rate')
    parser.add_argument('--lr_decay_rate', default=0.997, type=float, help='Learning rate decay rate')
    parser.add_argument('--weight_decay_rate', default=0, type=float, help='Weight decay rate')
    parser.add_argument('--decay_every', default=5, type=int, help='Number of epochs for each step lr decay')
    parser.add_argument('--beta_max', default=0.2, type=float, help='Maximum possible value for beta in VAE')
    parser.add_argument('--M', default=100, type=int, help='The number of epochs for an annealing cycle in VAE')
    parser.add_argument('--R', default=0.5, type=float, help='The portion of epochs within a cycle to increase beta in VAE')
    parser.add_argument('--model_type', default='bfm', choices=['ae', 'vae', 'bfm', 'iterative_svd', 'rbsvd', 'svd', 'als', 'ncf'], type=str, 
        help='Choice of model for collaborative filtering')
    parser.add_argument('--test_size', default=0.1, type=float, help='Validation set size')
    parser.add_argument('--bfm_use_iu', action='store_true', help='Whether to use implicit user features for BFM models')
    parser.add_argument('--bfm_use_ii', action='store_true', help='Whether to use implicit movie features for BFM models')
    parser.add_argument('--bfm_auxiliary_statistical_feature', action='store_true', help='Whether to use additional user and movie statistical features as auxiliary features for BFM models')
    parser.add_argument('--bfm_auxiliary_latent_code', action='store_true', help='Whether to use additional vae encoded latent vector as auxiliary features for BFM models. ' + 
        'If true, pre-trained VAE model path must be specified via ckpt_path')
    parser.add_argument('--min_rating', default=1, type=int, help='Minimum possible rating, used for clipping predictions')
    parser.add_argument('--max_rating', default=5, type=int, help='Maximum possible rating, used for clipping predictions')
    parser.add_argument('--rank_bfm', default=10, type=int, help='rank for BFM')
    parser.add_argument('--rank_rbsvd', default=12, type=int, help='rank for RBSVD')
    parser.add_argument('--rank_svd', default=9, type=int, help='rank for SVD')
    parser.add_argument('--rank_als', default=3, type=int, help='rank for ALS')
    parser.add_argument('--bfm_regressor', default='op', choices=['blr', 'op'], type=str, help='Choice of regressor for BFM')
    parser.add_argument('--lambda1', default=0.075, type=float, help='lambda1 for RBSVD')
    parser.add_argument('--lambda2', default=0.04, type=float, help='lambda2 for RBSVD')
    parser.add_argument('--lambda_als', default=0.2, type=float, help='lambda for ALS')
    parser.add_argument('--shrinkage', default=38, type=int, help='Shrinkage for Iterative SVD')
    parser.add_argument('--k_fold', default=10, type=int, help='The number of folds for cross validation.')
    parser.add_argument('--cv_folder', default='cv_results', type=str, help='Folder to store cross validation models and predictions')
    parser.add_argument('--model_types_blending', nargs='*', default=['ae', 'vae', 'bfm_base', 'bfm_base+implicit_blr', 'bfm_base+implicit_op', 'iterative_svd', 'rbsvd', 'ncf'], 
        help='All model types to be blended')
    parser.add_argument('--model_names_blending', nargs="*", default=['ae_cv', 'vae_cv', 'bfm_base_cv', 'bfm_base+implicit_blr_cv', 'implicit_op_cv', 'iterative_svd_cv', 'rbsvd_cv', 'ncf_cv'], 
        help='Model names for each of the model type. Length must match model_types_blending.')
    parser.add_argument('--blender_model_type', default='gb', choices=['lr', 'xgboost', 'gb'], type=str, 
        help='Choice of model for blending')
    parser.add_argument('--final_pred_names', nargs="*", default=['ae_final', 'vae_final', 'bfm_base_final', 'bfm_base+implicit_blr_final', 'bfm_base+implicit_op_final', 'iterative_svd_final', 'rbsvd_final', 'ncf_final'],
        help='Names of predictions for all final models')
    parser.add_argument('--blend_for_submission', action='store_true', help='Whether to blend all final predictions for submission. If true, final_pred_names must be specified')
    parser.add_argument('--user_code_length_ncf', default=128, type=int, help='users latent vector dimension for NCF')
    parser.add_argument('--movie_code_length_ncf', default=128, type=int, help='Movie latent vector dimension for NCF')
    parser.add_argument('--multiplication_code_length_ncf', default=128, type=int, help='Dimension of multiplcation feature for NCF')
    parser.add_argument('--feature_dropout_ncf', default=0.0, type=float, help='Feature dropout rate for NCF')
    parser.add_argument('--hidden_embeddingnet_ncf', nargs='*', default=[256, 128], 
        help='Hidden layers for the embeddingnet in NCF')
    parser.add_argument('--hidden_judgenet_ncf', nargs='*', default=[256, 128], 
        help='Hidden layers for the judgenet in NCF')
    parser.add_argument('--dropouts_embeddingnet_ncf', nargs='*', default=[0.05, 0.1, 0.1], 
        help='Dropout rates for layers in the embeddingnet in NCF')
    parser.add_argument('--dropouts_judgenet_ncf', nargs='*', default=[0.5, 0.25], 
        help='Dropout rates for layers in the judgenet in NCF')
    

    args = parser.parse_args(argv)

    args.hidden_embeddingnet_ncf = [int(x) for x in args.hidden_embeddingnet_ncf]
    args.hidden_judgenet_ncf = [int(x) for x in args.hidden_judgenet_ncf]
    args.dropouts_embeddingnet_ncf = [float(x) for x in args.dropouts_embeddingnet_ncf]
    args.dropouts_judgenet_ncf = [float(x) for x in args.dropouts_judgenet_ncf]

    if args.beta_max < 0  or args.beta_max > 1:
        parser.error('beta_max should be in [0, 1]!')

    values = [len(args.model_types_blending), len(args.model_names_blending)]
    if args.blend_for_submission:
        values.append(len(args.final_pred_names))
    if not all(v == values[0] for v in values):
        parser.error('Length of model_types_blending, model_names_blending, and final_pred_names (optional) must match!')

    args_dict = vars(args)

    return args_dict