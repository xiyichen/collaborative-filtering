import argparse
import torch

def parse_config(argv=None):
    parser = argparse.ArgumentParser(description='Parameters for collaborative filtering.')
    parser.add_argument('--final_model', action='store_true', help='Whether to train the model using the entire training set (for final submission)')
    parser.add_argument('--save_model', action='store_true', help='Whether to save trained model')
    parser.add_argument('--train_csv_path', default='./data_train.csv', type=str,
                        help='File path for the entire training set')
    parser.add_argument('--test_csv_path', default='./sampleSubmission.csv', type=str,
                        help='File path for the public test set (no ground truth)')
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
        help='Type of beta annealing schedule (or None to use a constant beta value, which is defined as beta_max)')
    parser.add_argument('--use_user_bias', action='store_true', help='Whether to use user bias')
    parser.add_argument('--use_movie_bias', action='store_true', help='Whether to use movie bias')
    parser.add_argument('--hidden_dimension', default=256, type=int, help='Number of neurons in the hidden layer for AE and VAE')
    parser.add_argument('--latent_dimension', default=32, type=int, help='Dimension of the latent space Z for AE and VAE')
    parser.add_argument('--dropout_p', default=0.5, type=float, help='Dropout probability for neural network models')
    parser.add_argument('--weight_init_type', default='xavier', choices=['xavier', 'kaiming', None], type=str, 
        help='Method to initialize layer weights for neural network models')
    parser.add_argument('--activation_type', default='tanh', choices=['tanh', 'sigmoid', 'leakyrelu', 'relu'], type=str, 
        help='Type of activation for neural network models')
    parser.add_argument('--batch_size', default=1024, type=int, help='Batch size for neural network models')
    parser.add_argument('--num_epochs', default=2000, type=int, help='Number of epochs to train the model')
    parser.add_argument('--last_epoch', default=-1, type=int, help='Last trained epoch, for resume training of a neural network model')
    parser.add_argument('--init_lr', default=0.025, type=float, help='Initial learning rate')
    parser.add_argument('--decay_rate', default=0.997, type=float, help='Decay rate for AE/VAE/RBSVD')
    parser.add_argument('--decay_every', default=5, type=int, help='Number of epochs for each step lr decay')
    parser.add_argument('--beta_max', default=0.2, type=float, help='Maximum possible value for beta in VAE')
    parser.add_argument('--M', default=100, type=int, help='The number of epochs for an annealing cycle in VAE')
    parser.add_argument('--R', default=0.5, type=float, help='The portion of epochs within a cycle to increase beta in VAE')
    parser.add_argument('--model_type', default='bfm', choices=['ae', 'vae', 'bfm', 'iterative_svd', 'rbsvd', 'svd', 'als', 'ncf'], type=str, 
        help='Choice of model for collaborative filtering')
    parser.add_argument('--test_size', default=0.1, type=float, help='Validation set size')
    parser.add_argument('--bfm_use_iu', action='store_true', help='Whether to use implicit user features for BFM models')
    parser.add_argument('--bfm_use_ii', action='store_true', help='Whether to use implicit movie features for BFM models')
    parser.add_argument('--bfm_auxiliary_statistical_feature', action='store_true', help='Whether to use additional user and movie statistical features as auxiliary features')
    parser.add_argument('--bfm_auxiliary_latent_code', action='store_true', help='Whether to use additional vae encoded latent vector as auxiliary features. ' + 
        'If true, pre-trained VAE model path must be specified via ckpt_path')
    parser.add_argument('--min_rating', default=1, type=int, help='Minimum possible rating, used for clipping predictions')
    parser.add_argument('--max_rating', default=5, type=int, help='Maximum possible rating, used for clipping predictions')
    parser.add_argument('--rank', default=10, type=int, help='rank for BFM/RBSVD')
    parser.add_argument('--bfm_regressor', default='op', choices=['blr', 'op'], type=str, help='Choice of regressor for BFM')
    parser.add_argument('--lambda1', default=0.075, type=float, help='lambda1 for RBSVD')
    parser.add_argument('--lambda2', default=0.04, type=float, help='lambda2 for RBSVD')
    
    args = parser.parse_args(argv)

    if args.beta_max < 0  or args.beta_max > 1:
        parser.error('beta_max should be in [0, 1]')

    args_dict = vars(args)

    return args_dict