from tqdm import tqdm
import numpy as np
from utils import *
import os
from sklearn.utils import shuffle

class RBSVD_trainer:
	def __init__(self, **args):
		'''
		Trainer for Regularized+Biased SVD, minimizing the objective using Stochastic Graident Descent (SGD).
		Refenrence: https://www.cs.uic.edu/~liub/KDD-cup-2007/proceedings/Regular-Paterek.pdf
		'''
		self.ckpt_folder = args.get('ckpt_folder')
		self.pred_folder = args.get('pred_folder')
		self.model_name = args.get('model_name')
		self.num_users = args.get('num_users')
		self.num_movies = args.get('num_movies')
		self.rank = args.get('rank_rbsvd')
		self.lambda1 = args.get('lambda1')
		self.lambda2 = args.get('lambda2')
		self.use_user_bias = args.get('use_user_bias')
		self.use_movie_bias = args.get('use_movie_bias')

	def train(self, users_train, movies_train, ratings_train, users_test=None, movies_test=None, ratings_test=None, **args):
		# Initialize components.
		self.U = np.random.uniform(0, 0.01, (self.num_users, self.rank))
		self.V = np.random.uniform(0, 0.01, (self.num_movies, self.rank))
		if self.use_user_bias:
			self.biasU = np.zeros(self.num_users)
		if self.use_movie_bias:
			self.biasV = np.zeros(self.num_movies)
		lr = args.get('init_lr')
		global_mean_ratings = np.mean(ratings_train)
		best_rmse = np.inf
		num_epochs = args.get('num_iterations')
		use_validation = not args.get('final_model')
		with tqdm(total=len(users_train) * num_epochs) as pbar:
			for epoch in range(1, num_epochs + 1):
				users_train_sh, movies_train_sh, ratings_train_sh = shuffle(users_train, movies_train, ratings_train)

				# Shuffle the dataset to perform SGD for the current epoch.
				for user, movie, rating in zip(users_train_sh, movies_train_sh, ratings_train_sh):
					# Get components for the current user and movie
					U_d = self.U[user, :]
					V_n = self.V[movie, :]
					if self.use_user_bias:
						biasU_d = self.biasU[user]
					if self.use_movie_bias:
						biasV_n = self.biasU[movie]
					pred = U_d.dot(V_n)
					if self.use_user_bias:
						pred += biasU_d
					if self.use_movie_bias:
						pred += biasV_n
					delta = rating - pred
					pbar.update(1)

					try:
						'''
						Gradient descent to update U, V and if using bias, biasU and biasV.
						'''
						new_U_d = U_d + lr * (delta * V_n - self.lambda1 * U_d)
						new_V_n = V_n + lr * (delta * U_d - self.lambda1 * V_n)
						if self.use_user_bias:
							lambda2_term = biasU_d - global_mean_ratings
							if self.use_movie_bias:
								lambda2_term += biasV_n
							new_biasU_d = biasU_d + lr * (delta - self.lambda2 * (lambda2_term))
						if self.use_movie_bias:
							lambda2_term = biasV_n - global_mean_ratings
							if self.use_user_bias:
								lambda2_term += biasU_d
							new_biasV_n = biasV_n + lr * (delta - self.lambda2 * (lambda2_term))
					except:
						continue

					self.U[user, :] = new_U_d
					self.V[movie, :] = new_V_n
					if self.use_user_bias:
						self.biasU[user] = new_biasU_d
					if self.use_movie_bias:
						self.biasV[movie] = new_biasV_n

				if use_validation:
					predictions = self.predict(users_test, movies_test)
					reconstruction_rmse = get_score(predictions, ratings_test)
					if reconstruction_rmse < best_rmse:
						best_rmse = reconstruction_rmse
					pbar.set_description('At epoch {:3d}: best loss {:.4f}, current loss {:.4f}'.format(epoch, best_rmse, reconstruction_rmse))

				if epoch % args.get('decay_every') == 0:
					lr /= args.get('decay_rate')

		# Save the model as components.
		if args.get('save_model'):
			np.savetxt(os.path.join('.', self.ckpt_folder, self.model_name + '_U.txt'), self.U)
			np.savetxt(os.path.join('.', self.ckpt_folder, self.model_name + '_V.txt'), self.V)
			if self.use_user_bias:
				np.savetxt(os.path.join('.', self.ckpt_folder, self.model_name + '_biasU.txt'), self.biasU)
			if self.use_movie_bias:
				np.savetxt(os.path.join('.', self.ckpt_folder, self.model_name + '_biasV.txt'), self.biasV)

		# Save prediction for either the full num_users*num_movies reconstruction matrix or only the user-movie indices in the test set.
		save_pred_type = args.get('save_pred_type')
		if save_pred_type is not None:
			if save_pred_type == 'full':
				full_matrix_users, full_matrix_movies = get_all_user_movie_indices(self.num_users, self.num_movies)
				pred = self.predict(full_matrix_users, full_matrix_movies, save_pred_type)
				np.savetxt(os.path.join('.', self.pred_folder, self.model_name + '_pred_full.txt'), pred)
			else:
				pred = self.predict(users_test, movies_test, save_pred_type)
				np.savetxt(os.path.join('.', self.pred_folder, self.model_name + '_pred_test.txt'), pred)

	def predict(self, users_test, movies_test, save_pred_type=None):
		preds = list()
		for user, movie in zip(users_test, movies_test):
			pred = self.U[user, :].dot(self.V[movie, :])
			if self.use_user_bias:
				pred += self.biasU[user]
			if self.use_movie_bias:
				pred += self.biasV[movie]
			preds.append(pred)
		preds = np.array(preds)

		if save_pred_type == 'full':
			preds = preds.reshape((self.num_users, self.num_movies))

		return preds