# An Ensemble Based Approach to Collaborative Filtering

Semester project for Computational Intelligence Lab by Xiyi Chen, Pengxi Liu, Jiasong Guo, Chuhao Feng.

## List parameters and descriptions
To see all available parameters and their descriptions, simply call:
```
python main.py --help
```

## Training
We provide examples for training each type of model, training a final model, cross validation, and blending. For each individual model, the parameters passed in are optimal. For more details on each model, check out our [report](./report.pdf). We provide a colab notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FAtMK_piAXSFwHRoO4SF5SrdyVEdiSNU?usp=sharing) that shows these training examples and results.

### SVD
```
python main.py --model_type 'svd' --model_name 'SVD_test' --rank_svd 9 --random_seed 42
```

### ALS
```
python main.py --model_type 'als' --model_name 'ALS_test' --rank_svd 9 --rank_als 3 --num_iterations 20 \
    --lambda_als 0.1 --random_seed 42 --save_pred_type 'full' --save_model
```

### Iterative SVD
```
python main.py --model_type 'iterative_svd' --model_name 'IterSVD_test' --shrinkage 38 \
    --num_iterations 15 --random_seed 42
```

### Regularized + Biased SVD
```
python main.py --model_type 'rbsvd' --model_name 'RBSVD_test' --num_iterations 50 --lambda1 0.075 --lambda2 0.04 \
    --init_lr 0.05 --lr_decay_rate 0.7 --decay_every 5 --rank_rbsvd 12 --use_user_bias --use_movie_bias --random_seed 42
```

### Bayesian FM Base (Gibbs)
```
python main.py --model_type 'bfm' --model_name 'BFM_base+implicit_op_test' \
    --num_iterations 500 --rank_bfm 10 --bfm_regressor 'blr' --random_seed 42
```

### Bayesian FM Base + Implicit (Gibbs)
```
python main.py --model_type 'bfm' --model_name 'BFM_base+implicit_op_test' --bfm_use_iu --bfm_use_ii \
    --num_iterations 500 --rank_bfm 10 --bfm_regressor 'blr' --random_seed 42
```

### Bayesian FM Base + Implicit (Ordinal Probit)
```
python main.py --model_type 'bfm' --model_name 'BFM_base+implicit_op_test' --bfm_use_iu --bfm_use_ii \
    --num_iterations 500 --rank_bfm 10 --bfm_regressor 'op' --random_seed 42
```

### Autoencoder
```
python main.py --model_type 'ae' --model_name 'AE_test' --lr_decay_rate 0.992 \
    --activation 'leakyrelu' --weight_init_type 'kaiming' --num_iterations 1000 --use_user_bias --random_seed 42 \
    --hidden_dimension 256 --latent_dimension 32
```

### Variational Autoencoder
```
python main.py --model_type 'vae' --model_name 'VAE_test' --lr_decay_rate 0.997 \
    --activation 'tanh' --weight_init_type 'xavier' --num_iterations 2000 --use_user_bias --random_seed 42 \
    --hidden_dimension 256 --latent_dimension 32 --beta_annealing_schedule 'cyclic' --beta_max 0.2 --M 100 \
    --R 0.5
```

### Neural Collaborative Filtering
```
python main.py --model_type ncf --model_name 'ncf_test' --init_lr 1e-3 --weight_decay_rate 1e-4 \
    --lr_decay_rate 0.97 --num_iterations 17 --batch_size 512 --user_code_length_ncf 128 \
    --movie_code_length_ncf 128 --multiplication_code_length_ncf 128 --feature_dropout_ncf 0.0 \
    --hidden_embeddingnet_ncf 256 128 --hidden_judgenet_ncf 256 128 \
    --dropouts_embeddingnet_ncf 0.05 0.1 0.1 --dropouts_judgenet_ncf 0.5 0.25
```

### Example: Train a final model with all available data for submission
```
python main.py --model_type 'vae' --model_name 'VAE_final' --lr_decay_rate 0.997 \
    --activation 'tanh' --weight_init_type 'xavier' --num_iterations 2000 --use_user_bias --random_seed 42 \
    --hidden_dimension 256 --latent_dimension 32 --beta_annealing_schedule 'cyclic' --beta_max 0.2 --M 100 \
    --R 0.5 --save_pred_type 'test_indices' --final_model
```

### Example: Cross Validation for VAE
```
python cross_validation.py --model_type 'vae' --model_name 'VAE_cv' --lr_decay_rate 0.997 \
    --activation 'tanh' --weight_init_type 'xavier' --num_iterations 2000 --use_user_bias --random_seed 42 \
    --hidden_dimension 256 --latent_dimension 32 --beta_annealing_schedule 'cyclic' --beta_max 0.2 --M 100 \
    --R 0.5 --save_model --save_pred_type 'full'
```

### Blending
To blend predictions, you will first need to do cross validation on models you would like to include. The prediction results of cross validation should be saved in `pred_folder`. When blending, the `model_names_blending` should contain the values you used as `model_name` in cross_validation.py for each model. The `final_pred_names` should contain the values you used as `model_name` when training the final model for each method.

The following is an example to blend two methods: BFM Base + Implicit (Ordinal Probit) and VAE.
```
python blending.py --model_name 'blended_final' --k_fold 10 --random_seed 42 --blender_model_type 'gb' \
    --model_types_blending 'bfm_base+implicit_op' 'vae' --model_names_blending 'BFM_Ordered_Probit_SVD++' 'VAE' \
    --final_pred_names 'BFM_OrderedProbit_full' 'VAE_full' --blend_for_submission
```
The directory structure should be:
```
collaborative-filtering
├── $pred_folder
|   ├── BFM_Ordered_Probit_SVD++_fold_*.txt
|   ├── VAE_fold_*.txt
|   ├── BFM_OrderedProbit_full.txt
|   ├── VAE_full.txt
```
The following example blends all models but the two baseline methods (8 in total), which produces the lowest local CV score out of all models we tested. We refer to this model as `Blending (Gradient Boosting)` in the experiments.
```
python blending.py --model_name 'blended_final' --k_fold 10 --random_seed 42 --blender_model_type 'gb' \
    --model_types_blending 'ae' 'vae' 'bfm_base' 'bfm_base+implicit_blr' \ 
    'bfm_base+implicit_op' 'iterative_svd' 'rbsvd' 'ncf' \
    --model_names_blending 'AE_cv' 'VAE_cv' 'BFM_base_cv' 'BFM_SVD++_cv' 'BFM_Ordered_Probit_SVD++_cv' \ 
    'IterSVD_cv' 'RBSVD_cv' 'NCF_cv' \ 
    --final_pred_names 'AE_full' 'VAE_full' 'BFM_base_full' 'BFM_SVD++_full' 'BFM_OrderedProbit_full' \ 
    'IterSVD_full' 'RBSVD_full' 'NCF_full' --blend_for_submission
```

## Generating predictions
To generate prediction files for single models, add flag `--save_pred_type` which is either 'full' (for full reconstructed 10000x1000 matrix) or 'test_indices' (only saving predictions for the test user and movies). To train a final model with all available training data and generate predictions for the test set, you can use `--final_model --save_pred_type 'test_indices'`.

To generate prediction files for blending multiple models, add flag `--blend_for_submission`.

## Generate .csv file for submission
To generate a csv file for submission, specify `save_pred_type` and `model_name`
For example:
```
python submission.py --save_pred_type 'test_indices' --model_name 'blended_final'
```
reads `blended_final.txt` from `pred_folder` and extract user and movie indices of the test set, and generate a csv file for submission.

## Experiment results
Below are our experiment results. You can reproduce them by using random seed 42 for cross validation.
| Method                                       | Local CV Score | Public Test Score |
| :------------------------------------------- |  :-----------: |  :--------------: |
| SVD (baseline 1)                             |     1.0081     |       1.0049      |
| ALS (baseline 1)                             |     0.9899     |       0.9874      |
| Iterative SVD                                |     0.9840     |       0.9816      |
| Regularized + Biase SVD                      |     0.9820     |       0.9788      |
| Bayesian FM (Gibbs)                          |     0.9779     |       0.9749      |
| Bayesian FM Base + Implicit (Gibbs)          |     0.9715     |       0.9687      |
| Bayesian FM Base + Implicit (Ordinal Probit) |     0.9694     |       0.9672      |
| Autoencoder                                  |     0.9791     |       0.9758      |
| Variational Autoencoder                      |     0.9769     |       0.9749      |
| Neural Collaborative Filtering               |     0.9889     |       0.9856      |
| Blending (Linear Regression)                 |     0.9682     |       0.9659      |
| Blending (XGBoost)                           |     0.9676     |     **0.9656**    |
| Blending (Gradient Boosting)                 |   **0.9674**   |     **0.9656**    |

