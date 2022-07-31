from models.NCF import NCF
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
from utils import *
import os

class MyDataset(Dataset):
	def __init__(self, dataset, target):
		self.dataset = dataset
		self.target = target
	def __len__(self):
		return self.dataset.shape[0]
	def __getitem__(self, idx):
		return self.dataset[idx], self.target[idx]

class NCF_trainer:
	def __init__(self, **args):
		self.ckpt_folder = args.get('ckpt_folder')
		self.pred_folder = args.get('pred_folder')
		self.model_name = args.get('model_name')
		self.min_rating = args.get('min_rating')
		self.max_rating = args.get('max_rating')
		movie_code_length = 128
		user_code_length = 128
		multiplication_length = 128
		self.device = args.get('device')
		self.model = NCF(**args).to(self.device)
		if not '_fold_' in self.model_name:
			print(self.model)
		ckpt_path = args.get('ckpt_path')
		if ckpt_path is not None:
			self.model.load_state_dict(torch.load(ckpt_path, map_location=args.get('device')))

	def train(self, users_train, movies_train, ratings_train, users_test=None, movies_test=None, ratings_test=None, **args):
		batch_size = args.get('batch_size')
		# Normalize labels
		ratings_train_normalized, ratings_train_mean, ratings_train_std = stand_norm(ratings_train)
		# Build dataloaders
		train_dataset = MyDataset(np.vstack((users_train, movies_train)).T, ratings_train_normalized.reshape(-1, 1))
		train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
		use_validation = not args.get('final_model')
		if use_validation:
			ratings_test_normalized = (ratings_test - ratings_train_mean) / ratings_train_std
			test_dataset = MyDataset(np.vstack((users_test, movies_test)).T, ratings_test_normalized.reshape(-1, 1))
			test_dataloader = DataLoader(dataset=test_dataset, batch_size=len(users_test), shuffle=True)
		best_rmse = np.inf
		num_epochs = args.get('num_iterations')
		optimizer = optim.Adam(self.model.parameters(), lr=args.get('init_lr'), weight_decay=args.get('weight_decay_rate'))
		scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.get('lr_decay_rate'), last_epoch=args.get('last_iteration'))
		criterion = nn.MSELoss(reduction='mean')
		with tqdm(total=len(train_dataloader) * num_epochs) as pbar:
			for epoch in range(num_epochs):
				for idx, (dataset_batch, target_batch) in enumerate(train_dataloader):
					optimizer.zero_grad()
					dataset_batch = dataset_batch.to(self.device)
					target_batch = target_batch.to(self.device).float()
					pred = self.model(dataset_batch[:, 0], dataset_batch[:, 1])
					loss = criterion(pred, target_batch)
					loss.backward()
					optimizer.step()
					pbar.update(1)

				if use_validation:
					self.model.eval()
					for idx, (dataset_batch, target_batch) in enumerate(test_dataloader):
						dataset_batch = dataset_batch.to(self.device)
						target_batch = target_batch.to(self.device).float()
						# Denormalize labels before calculating validation RMSE
						pred = self.model(dataset_batch[:, 0], dataset_batch[:, 1])
						pred_denorm = de_norm(pred.squeeze(1), ratings_train_mean, ratings_train_std).detach().cpu().numpy()
						target_denorm = de_norm(target_batch.squeeze(1), ratings_train_mean, ratings_train_std).detach().cpu().numpy()
						reconstruction_rmse = get_score(pred_denorm, target_denorm)
						if reconstruction_rmse < best_rmse:
							best_rmse = reconstruction_rmse
						pbar.set_description('At epoch {:3d}: best loss {:.4f}, current loss {:.4f}, current LR {:.5f}'.format(epoch, best_rmse, reconstruction_rmse, optimizer.param_groups[0]["lr"]))

					self.model.train()
				scheduler.step()
		
		self.model.eval()
		# Save the model.
		if args.get('save_model'):
			torch.save(self.model.state_dict(), os.path.join('.', self.ckpt_folder, self.model_name + '.pt'))

		# Save prediction for either the full num_users*num_movies reconstruction matrix or only the user-movie indices in the test set.
		save_pred_type = args.get('save_pred_type')
		if save_pred_type is not None:
			if save_pred_type == 'full':
				full_matrix_users, full_matrix_movies = get_all_user_movie_indices(self.num_users, self.num_movies)
				pred = self.predict(torch.tensor(np.vstack((full_matrix_users, full_matrix_movies)).T), ratings_train_mean, ratings_train_std).reshape(args.get('num_users'), args.get('num_movies'))
				np.savetxt(os.path.join('.', self.pred_folder, self.model_name + '_pred_full.txt'), pred)
			else:
				pred = self.predict(torch.tensor(np.vstack((users_test, movies_test)).T), ratings_train_mean, ratings_train_std)
				np.savetxt(os.path.join('.', self.pred_folder, self.model_name + '_pred_test.txt'), pred)


	def predict(self, dataset_torch, ratings_train_mean, ratings_train_std):
		pred = self.model(dataset_torch[:, 0], dataset_torch[:, 1])
		pred_denorm = de_norm(pred.squeeze(1), ratings_train_mean, ratings_train_std).detach().cpu().numpy().clip(self.min_rating, self.max_rating)
		return pred_denorm