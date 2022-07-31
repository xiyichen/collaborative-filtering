from tqdm import tqdm
import numpy as np
from utils import *
import os

class SVD_trainer:
	def __init__(self, **args):
		'''
		Trainer for Singular Value Decomposition (SVD)
		'''
		self.ckpt_folder = args.get('ckpt_folder')
		self.pred_folder = args.get('pred_folder')
		self.model_name = args.get('model_name')
		self.num_movies = args.get('num_movies')
		self.rank = args.get('rank')

	def SVD(self, A, num_movies, k=9):
		U, s, Vt = np.linalg.svd(A, full_matrices=False)

		# using the top k eigenvalues
		S = np.zeros((num_movies, num_movies))
		S[:k, :k] = np.diag(s[:k])

		# reconstruc matrix
		return U, S, Vt

	def train(self, users_train, movies_train, ratings_train, users_test=None, movies_test=None, ratings_test=None, **args):
		_, data_torch, mask_torch, _ = get_dataloader(users_train, movies_train, ratings_train, impute=np.nan, **args)
		# Perform standardization on the user-movie matrix A and zero-impute missing values.
		A = data_torch.detach().cpu().numpy()
		A, col_mean, col_std = stand_norm(A)
		A = impute_matrix(A)
		# Perform SVD.
		U, S, Vt = self.SVD(A, self.num_movies, self.rank)
		# Initialize components.
		self.U = U
		self.S = S
		self.Vt = Vt
		use_validation = not args.get('final_model')

		reconstructed_matrix = self.predict(col_mean, col_std)
		if use_validation:
			predictions = extract_prediction_from_full_matrix(reconstructed_matrix, users_test, movies_test)
			print('Validation loss: {:.4f}'.format(get_score(predictions, ratings_test)))

		# Save the model as components.
		if args.get('save_model'):
			np.savetxt(os.path.join('.', self.ckpt_folder, self.model_name + '_U.txt'), self.U)
			np.savetxt(os.path.join('.', self.ckpt_folder, self.model_name + '_S.txt'), self.S)
			np.savetxt(os.path.join('.', self.ckpt_folder, self.model_name + '_Vt.txt'), self.Vt)

		# Save prediction for either the full num_users*num_movies reconstruction matrix or only the user-movie indices in the test set.
		save_pred_type = args.get('save_pred_type')
		if save_pred_type is not None:
			if save_pred_type == 'full':
				np.savetxt(os.path.join('.', self.pred_folder, self.model_name + '_pred_full.txt'), reconstructed_matrix)
			else:
				predictions = extract_prediction_from_full_matrix(reconstructed_matrix, users_test, movies_test)
				np.savetxt(os.path.join('.', self.pred_folder, self.model_name + '_pred_test.txt'), predictions)


	def predict(self, col_mean, col_std):
		return de_norm(self.U.dot(self.S).dot(self.Vt), col_mean, col_std)
		