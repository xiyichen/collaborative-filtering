from tqdm import tqdm
import numpy as np
from utils import *
import os

class IterativeSVD_trainer:
	def __init__(self, **args):
		'''
		Trainer for Iterative SVD
		'''
		self.ckpt_folder = args.get('ckpt_folder')
		self.pred_folder = args.get('pred_folder')
		self.model_name = args.get('model_name')
		self.num_iterations = args.get('num_iterations')
		self.shrinkage = args.get('shrinkage')
		self.min_rating = args.get('min_rating')
		self.max_rating = args.get('max_rating')

	def IterSVD(self, A, mask_A, shrinkage=38, n_itr=15):
		X = A.copy()
		for i in range(n_itr):
			# SVD
			U, s, Vt = np.linalg.svd(X, full_matrices=False)
			# truncate SVD
			s_ = (s - shrinkage).clip(min=0)
			# reconstruct
			X = U.dot(np.diag(s_)).dot(Vt)
			# restore observed ratings
			X[mask_A] = A[mask_A]
			print("%sth iteration is complete." % i)

		Arc = X.clip(self.min_rating, self.max_rating)
		return Arc

	def train(self, users_train, movies_train, ratings_train, users_test=None, movies_test=None, ratings_test=None, **args):
		_, data_torch, mask_torch, _ = get_dataloader(users_train, movies_train, ratings_train, impute=np.nan, **args)
		# Perform standardization on the user-movie matrix A and zero-impute missing values.
		A = data_torch.detach().cpu().numpy()
		mask_A = ~np.isnan(A)
		A = impute_matrix(A, 'col_mean')
		# Perform SVD.
		A_sub = self.IterSVD(A, mask_A, self.shrinkage, self.num_iterations)
		use_validation = not args.get('final_model')

		reconstructed_matrix = A_sub
		if use_validation:
			predictions = extract_prediction_from_full_matrix(reconstructed_matrix, users_test, movies_test)
			print('Validation loss: {:.4f}'.format(get_score(predictions, ratings_test)))

		# Save the model as components.
		if args.get('save_model'):
			pass

		# Save prediction for either the full num_users*num_movies reconstruction matrix or only the user-movie indices in the test set.
		save_pred_type = args.get('save_pred_type')
		if save_pred_type is not None:
			if save_pred_type == 'full':
				np.savetxt(os.path.join('.', self.pred_folder, self.model_name + '_pred_full.txt'), reconstructed_matrix)
			else:
				predictions = extract_prediction_from_full_matrix(reconstructed_matrix, users_test, movies_test)
				np.savetxt(os.path.join('.', self.pred_folder, self.model_name + '_pred_test.txt'), predictions)


	def predict(self, users_test, movies_test, save_pred_type=None):
		pass
