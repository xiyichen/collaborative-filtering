from models.AE import AE
from tqdm import tqdm
import torch
import numpy as np
from utils import *
import os
import torch.optim as optim

class AE_trainer:
	def __init__(self, **args):
		self.ckpt_folder = args.get('ckpt_folder')
		self.pred_folder = args.get('pred_folder')
		self.model_name = args.get('model_name')
		self.ckpt_path = args.get('ckpt_path')
		self.min_rating = args.get('min_rating')
		self.max_rating = args.get('max_rating')
		device = args.get('device')
		self.model = AE(**args).to(device)
		if not '_fold_' in self.model_name:
			print(self.model)
		ckpt_path = args.get('ckpt_path')
		if ckpt_path is not None:
			self.model.load_state_dict(torch.load(ckpt_path))

	def train(self, users_train, movies_train, ratings_train, users_test=None, movies_test=None, ratings_test=None, **args):
		dataloader, data_torch, mask_torch, user_id_torch = get_dataloader(users_train, movies_train, ratings_train, **args)
		optimizer = optim.Adam(self.model.parameters(), lr=args.get('init_lr'))
		scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.get('decay_rate'), last_epoch=args.get('last_iteration'))
		best_rmse = np.inf
		num_epochs = args.get('num_iterations')
		use_validation = not args.get('final_model')
		with tqdm(total=len(dataloader) * num_epochs) as pbar:
			for epoch in range(num_epochs):
				for idx, (data_batch, mask_batch, user_id_batch) in enumerate(dataloader):
					optimizer.zero_grad()
					reconstructed_batch, _ = self.model(data_batch, user_id_batch)
					loss = self.model.loss_function(data_batch, reconstructed_batch, mask_batch)

					loss.backward()
					optimizer.step()
					pbar.update(1)

				if use_validation:
					self.model.eval()
					reconstructed_matrix = self.predict(data_torch, user_id_torch)
					predictions = extract_prediction_from_full_matrix(reconstructed_matrix, users_test, movies_test)
					reconstruction_rmse = get_score(predictions, ratings_test)

					if reconstruction_rmse < best_rmse:
						best_rmse = reconstruction_rmse
					pbar.set_description('At epoch {:3d}: best loss {:.4f}, current loss {:.4f}, current LR {:.5f}'
						.format(epoch, best_rmse, reconstruction_rmse, optimizer.param_groups[0]["lr"]))
					self.model.train()
				scheduler.step()

		# Save the model.
		if args.get('save_model'):
			torch.save(self.model.state_dict(), os.path.join('.', self.ckpt_folder, self.model_name + '.pt'))

		# Save prediction for either the full num_users*num_movies reconstruction matrix or only the user-movie indices in the test set.
		save_pred_type = args.get('save_pred_type')
		if save_pred_type is not None:
			reconstructed_matrix = self.predict(data_torch, user_id_torch)
			if save_pred_type == 'full':
				np.savetxt(os.path.join('.', self.pred_folder, self.model_name + '_pred_full.txt'), reconstructed_matrix)
			else:
				predictions = extract_prediction_from_full_matrix(reconstructed_matrix, users_test, movies_test)
				np.savetxt(os.path.join('.', self.pred_folder, self.model_name + '_pred_test.txt'), predictions)

	def predict(self, data, user_id):
		data_reconstructed, _ = self.model(data, user_id)

		return data_reconstructed.detach().cpu().numpy().clip(self.min_rating, self.max_rating)