# An Ensemble Based Approach to Collaborative Filtering
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FAtMK_piAXSFwHRoO4SF5SrdyVEdiSNU?usp=sharing)

Semester project for Computational Intelligence Lab by Xiyi Chen, Pengxi Liu, Jiasong Guo, Chuhao Feng.

| Method                                       | Local CV Score | Public Test Score |
| :------------------------------------------- |  :-----------: |  :--------------: |
| SVD (baseline 1)                             |     1.0081     |       1.0049      |
| ALS (baseline 1)                             |     0.9899     |       0.9874      |
| Iterative SVD                                |     0.9840     |       0.9816      |
| Regularized + Biase SVD                      |     0.9820     |       0.9788      |
| Bayesian FM (Gibbs)                          |     0.9779     |       0.9749      |
| Bayesian FM Base + Implicit (Gibbs)          |     0.9715     |       0.9687      |
| Bayesian FM Base + Implicit (Ordinal Probit) |     0.9694     |       0.9672      |
| AE                                           |     0.9791     |       0.9758      |
| VAE                                          |     0.9769     |       0.9749      |
| Neural Collaborative Filtering               |     0.9889     |       0.9856      |
| Blending (Linear Regression)                 |     0.9682     |       0.9659      |
| Blending (XGBoost)                           |     0.9676     |     **0.9656**    |
| Blending (Gradient Boosting)                 |   **0.9674**   |     **0.9656**    |
