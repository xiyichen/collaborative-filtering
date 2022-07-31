from tqdm import tqdm
import torch
import numpy as np
from utils import extract_prediction_from_full_matrix, get_dataloader, get_all_user_movie_indices
from torch.utils.data import DataLoader, TensorDataset
import os
from typing import Dict, List, Union
from scipy import sparse as sps
from myfm import MyFMGibbsRegressor, MyFMOrderedProbit, RelationBlock
from myfm.utils.encoders import CategoryValueToSparseEncoder
from myfm.utils.callbacks import RegressionCallback, OrderedProbitCallback
import pickle
from models.VAE import VAE

class BFM_trainer:
	def __init__(self, **args):
		'''
		Bayesian Factorization Machine for Collaborative Filtering.
		Reference: https://github.com/tohtsky/myFM/tree/main/examples
		'''
		self.ckpt_folder = args.get('ckpt_folder')
		self.pred_folder = args.get('pred_folder')
		self.model_name = args.get('model_name')
		self.rank = args.get('rank')
		self.num_iterations = args.get('num_epochs')
		self.use_iu = args.get('bfm_use_iu')
		self.use_ii = args.get('bfm_use_ii')
		self.min_rating = args.get('min_rating')
		self.max_rating = args.get('max_rating')
		self.num_users = args.get('num_users')
		self.num_movies = args.get('num_movies')

	def augment_user_id(self, user_ids: List[int], user_vs_watched, user_to_internal, movie_to_internal) -> sps.csr_matrix:
		X = user_to_internal.to_sparse(user_ids)
		if not self.use_iu:
			return X
		data: List[float] = []
		row: List[int] = []
		col: List[int] = []
		for index, user_id in enumerate(user_ids):
			watched_movies = user_vs_watched.get(user_id, [])
			normalizer = 1 / max(len(watched_movies), 1) ** 0.5
			for mid in watched_movies:
				data.append(normalizer)
				col.append(movie_to_internal[mid])
				row.append(index)
		return sps.hstack(
			[
				X,
				sps.csr_matrix(
					(data, (row, col)),
					shape=(len(user_ids), len(movie_to_internal)),
				),
			],
			format="csr",
		)

	def augment_movie_id(self, movie_ids: List[int], movie_vs_watched, user_to_internal, movie_to_internal) -> sps.csr_matrix:
		X = movie_to_internal.to_sparse(movie_ids)
		if not self.use_ii:
			return X

		data: List[float] = []
		row: List[int] = []
		col: List[int] = []

		for index, movie_id in enumerate(movie_ids):
			watched_users = movie_vs_watched.get(movie_id, [])
			normalizer = 1 / max(len(watched_users), 1) ** 0.5
			for uid in watched_users:
				data.append(normalizer)
				row.append(index)
				col.append(user_to_internal[uid])
		return sps.hstack(
			[
				X,
				sps.csr_matrix(
					(data, (row, col)),
					shape=(len(movie_ids), len(user_to_internal)),
				),
			]
		)

	def get_auxiliary_features(self, X_train, X_test, users_train, movies_train, ratings_train, users_test, movies_test, **args):
		_, data_torch, mask_torch, user_id_torch = get_dataloader(users_train, movies_train, ratings_train, **args)

		if args.get('bfm_auxiliary_latent_code'):
			vae = VAE(**args).to(args.get('device'))
			vae.load_state_dict(torch.load(args.get('ckpt_path'), map_location=args.get('device')))
			vae.eval()
			_, z, _, _ = vae(data_torch, user_id_torch)
			z = z.cpu().detach().numpy()
			z_train = z[users_train]
			z_test = z[users_test]
			X_train = z_train
			X_test = z_test

		if args.get('bfm_auxiliary_statistical_feature'):
			data_np = data_torch.detach().cpu().numpy()
			mask = data_np != 0
			input_nans = np.where(mask, data_np, np.nan)
			users_mean = np.nanmean(input_nans, axis=1)
			users_std = np.nanstd(input_nans, axis=1)
			movies_mean = np.nanmean(input_nans, axis=0)
			movies_std = np.nanstd(input_nans, axis=0)

			users_mean_train = np.expand_dims(users_mean[users_train], axis=1)
			users_mean_test = np.expand_dims(users_mean[users_test], axis=1)
			users_std_train = np.expand_dims(users_std[users_train], axis=1)
			users_std_test = np.expand_dims(users_std[users_test], axis=1)
			movies_mean_train = np.expand_dims(movies_mean[movies_train], axis=1)
			movies_mean_test= np.expand_dims(movies_mean[movies_test], axis=1)
			movies_std_train = np.expand_dims(movies_std[movies_train], axis=1)
			movies_std_test= np.expand_dims(movies_std[movies_test], axis=1)

			X_train_aug = np.concatenate((users_mean_train, users_std_train, movies_mean_train, movies_std_train), axis=1)
			X_test_aug = np.concatenate((users_mean_test, users_std_test, movies_mean_test, movies_std_test), axis=1)
			if X_train is None:
				X_train = X_train_aug
			else:
				X_train = np.concatenate((X_train, X_train_aug), axis=1)
			if X_test is None:
				X_test = X_test_aug
			else:
				X_test = np.concatenate((X_test, X_test_aug), axis=1)

		return X_train, X_test

	def train(self, users_train, movies_train, ratings_train, users_test=None, movies_test=None, ratings_test=None, **args):
		X_train, X_test = None, None
		X_train, X_test = self.get_auxiliary_features(X_train, X_test, users_train, movies_train, ratings_train, users_test, movies_test, **args)

		use_validation = not args.get('final_model')
		user_to_internal = CategoryValueToSparseEncoder[int](
			users_train
		)
		movie_to_internal = CategoryValueToSparseEncoder[int](
			movies_train
		)
		movie_vs_watched: Dict[int, List[int]] = dict()
		user_vs_watched: Dict[int, List[int]] = dict()

		for (user_id, movie_id) in zip(users_train, movies_train):
			movie_vs_watched.setdefault(movie_id, list()).append(user_id)
			user_vs_watched.setdefault(user_id, list()).append(movie_id)
		feature_group_sizes = []

		feature_group_sizes.append(len(user_to_internal))  # user ids

		if self.use_iu:
			# all movies which a user watched
			feature_group_sizes.append(len(movie_to_internal))

		feature_group_sizes.append(len(movie_to_internal))  # movie ids

		if self.use_ii:
			# all the users who watched a movies
			feature_group_sizes.append(len(user_to_internal))

		grouping = [i for i, size in enumerate(feature_group_sizes) for _ in range(size)]
		train_blocks: List[RelationBlock] = []
		test_blocks: List[RelationBlock] = []
		for user_ids, movie_ids, blocks in [(users_train, movies_train, train_blocks), (users_test, movies_test, test_blocks)]:
			unique_users, user_map = np.unique(user_ids, return_inverse=True)
			blocks.append(RelationBlock(user_map, self.augment_user_id(unique_users, user_vs_watched, user_to_internal, movie_to_internal)))
			unique_movies, movie_map = np.unique(movie_ids, return_inverse=True)
			blocks.append(RelationBlock(movie_map, self.augment_movie_id(unique_movies, movie_vs_watched, user_to_internal, movie_to_internal)))
		callback = None
		regressor_type = args.get('bfm_regressor')
		if regressor_type == 'op':
			self.model = MyFMOrderedProbit(rank=self.rank)
			if use_validation:
				callback = OrderedProbitCallback(
					n_iter=self.num_iterations,
					X_test=X_test,
					y_test=ratings_test-1,
					X_rel_test=test_blocks,
					n_class=self.max_rating - self.min_rating + 1
				)
		else:
			self.model = MyFMGibbsRegressor(rank=self.rank)
			if use_validation:
				callback = RegressionCallback(
					n_iter=self.num_iterations,
					X_test=X_test,
					y_test=ratings_test-1,
					X_rel_test=test_blocks,
					clip_min=self.min_rating,
					clip_max=self.max_rating
				)

		self.model.fit(
			X_train,
			ratings_train-1,
			X_rel=train_blocks,
			grouping=grouping,
			n_iter=self.num_iterations,
			n_kept_samples=self.num_iterations,
			callback=callback
		)

		if args.get('save_model'):
			with open(os.path.join('.', self.ckpt_folder, self.model_name + '.pkl'), 'wb') as f:
				pickle.dump(self.model, f)

		save_pred_type = args.get('save_pred_type')
		if save_pred_type is not None:
			if save_pred_type == 'full':
				full_matrix_users, full_matrix_movies = get_all_user_movie_indices(self.num_users, self.num_movies)
				test_blocks_full: List[RelationBlock] = []
				unique_users, user_map = np.unique(np.array(full_matrix_users), return_inverse=True)
				test_blocks_full.append(RelationBlock(user_map, self.augment_user_id(unique_users, user_vs_watched, user_to_internal, movie_to_internal)))
				unique_movies, movie_map = np.unique(np.array(full_matrix_movies), return_inverse=True)
				test_blocks_full.append(RelationBlock(movie_map, self.augment_movie_id(unique_movies, movie_vs_watched, user_to_internal, movie_to_internal)))
				pred = self.predict(test_blocks_full, regressor_type, save_pred_type)
				np.savetxt(os.path.join('.', self.pred_folder, self.model_name + '_pred_full.txt'), pred)
			else:
				pred = self.predict(test_blocks, regressor_type, save_pred_type)
				np.savetxt(os.path.join('.', self.pred_folder, self.model_name + '_pred_test.txt'), pred)


	def predict(self, test_blocks, regressor_type, save_pred_type):
		if regressor_type == 'op':
			preds = self.model.predict_proba(None, X_rel = test_blocks).dot(np.arange(self.min_rating, self.max_rating+1)).clip(self.min_rating, self.max_rating)
		else:
			preds = self.model.predict(None, X_rel = test_blocks).clip(self.min_rating, self.max_rating)

		if save_pred_type == 'full':
			preds = preds.reshape((self.num_users, self.num_movies))

		return preds