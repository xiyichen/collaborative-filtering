from tqdm import tqdm
import numpy as np
from utils import *
import os

class ALS_trainer:
	def __init__(self, **args):
		'''
		Trainer for Alternating Least Squares (ALS)
		'''
		self.ckpt_folder = args.get('ckpt_folder')
		self.pred_folder = args.get('pred_folder')
		self.model_name = args.get('model_name')
		self.num_movies = args.get('num_movies')
		self.rank_svd = args.get('rank_svd')
		self.rank_als = args.get('rank_als')
		self.num_iterations = args.get('num_iterations')
		self.lambda_als = args.get('lambda_als')

	def ALS(self, A, mask_A, k=3, n_itr=20, lambda_=0.1):
		print("Initializing ALS")
		n, m = A.shape
		U, S, Vt = self.SVD(A, self.num_movies, self.rank_svd)
		U = np.copy(U[:,:k])
		V = np.copy(Vt[:k,:])

		print("Starting Iterations")
		# code for updating U and V is adapted from
		# https://github.com/mickeykedia/Matrix-Factorization-ALS
		for iter in range(n_itr):
			for i, Ri in enumerate(mask_A):
				U[i] = np.linalg.solve(np.dot(V, np.dot(np.diag(Ri), V.T)) + lambda_ * np.eye(k),
					np.dot(V, np.dot(np.diag(Ri), A[i].T))).T
			print("Error after solving for U matrix:", np.sum((mask_A * (A - np.dot(U, V))) ** 2) / np.sum(mask_A))

			for j, Rj in enumerate(mask_A.T):
				V[:,j] = np.linalg.solve(np.dot(U.T, np.dot(np.diag(Rj), U)) + lambda_ * np.eye(k),
					np.dot(U.T, np.dot(np.diag(Rj), A[:, j])))
			print("Error after solving for V matrix:", np.sum((mask_A * (A - np.dot(U, V))) ** 2) / np.sum(mask_A))
			print("%sth iteration is complete." % iter)

		return U, V

	def SVD(self, A, num_movies, k=9):
		U, s, Vt = np.linalg.svd(A, full_matrices=False)

		# using the top k eigenvalues
		S = np.zeros((num_movies, num_movies))
		S[:k, :k] = np.diag(s[:k])

		# reconstruct matrix
		return U, S, Vt

	def train(self, users_train, movies_train, ratings_train, users_test=None, movies_test=None, ratings_test=None, **args):
		_, data_torch, mask_torch, _ = get_dataloader(users_train, movies_train, ratings_train, impute=np.nan, **args)
		# Perform standardization on the user-movie matrix A and zero-impute missing values.
		A = data_torch.detach().cpu().numpy()
		mask_A = ~np.isnan(A)
		A, col_mean, col_std = stand_norm(A)
		A = impute_matrix(A)
		# Perform ALS.
		U, V = self.ALS(A, mask_A, self.rank_als, self.num_iterations, self.lambda_als)
		self.U = U
		self.V = V
		use_validation = not args.get('final_model')

		reconstructed_matrix = self.predict(col_mean, col_std)
		if use_validation:
			predictions = extract_prediction_from_full_matrix(reconstructed_matrix, users_test, movies_test)
			print('Validation loss: {:.4f}'.format(get_score(predictions, ratings_test)))

		# Save the model as components.
		if args.get('save_model'):
			np.savetxt(os.path.join('.', self.ckpt_folder, self.model_name + '_U.txt'), self.U)
			np.savetxt(os.path.join('.', self.ckpt_folder, self.model_name + '_V.txt'), self.V)

		# Save prediction for either the full num_users*num_movies reconstruction matrix or only the user-movie indices in the test set.
		save_pred_type = args.get('save_pred_type')
		if save_pred_type is not None:
			if save_pred_type == 'full':
				np.savetxt(os.path.join('.', self.pred_folder, self.model_name + '_pred_full.txt'), reconstructed_matrix)
			else:
				predictions = extract_prediction_from_full_matrix(reconstructed_matrix, users_test, movies_test)
				np.savetxt(os.path.join('.', self.pred_folder, self.model_name + '_pred_test.txt'), predictions)


	def predict(self, col_mean, col_std):
		return de_norm(self.U.dot(self.V), col_mean, col_std)
		