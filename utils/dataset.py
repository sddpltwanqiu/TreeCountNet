import os
import numpy as np
import cv2
import torch
import tifffile
import torch.utils.data as data

from utils.gaussian import choose_map_method


def make_dataset(root):
	file_path = []
	img = root + '//IMG'
	gt = root + '//GT'
	name_list = os.listdir(img)
	for name in name_list:
		imgs_path = os.path.join(img, name)
		gt_path = os.path.join(gt, name)
		file_path.append((imgs_path, gt_path))
	return file_path


class make_Dataset(data.Dataset):
	def __init__(self, root, arg, transform=None, target_transform=None):
		file_path = make_dataset(root)
		self.file_path = file_path
		self.transform = transform
		self.target_transform = target_transform
		self.gauss_kernel = arg.gauss_kernel
		self.sigma = arg.sigma
		self.factor = arg.factor
		self.adaption_gaussian = arg.adaption_gaussian
	
	def __getitem__(self, index):
		x_path, y_path = self.file_path[index]
		img_x = cv2.imread(x_path).astype(np.uint8)
		img_y = cv2.imread(y_path, flags=-1)
		img_y = choose_map_method(img_y, self.gauss_kernel, self.sigma, self.adaption_gaussian)
		#img_y = img_y * int(self.factor)
		
		if self.transform is not None:
			img_x = self.transform(img_x)
		if self.target_transform is not None:
			img_y = torch.Tensor(img_y/1.0)
			img_y = img_y.unsqueeze(0)
		return img_x, img_y
	
	def __len__(self):
		return len(self.file_path)
