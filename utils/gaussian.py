import cv2
import numpy as np
import scipy
import scipy.spatial


def choose_map_method(input_gt, gaussian_kernel, sigma, adaptive_kernel=False):
	if adaptive_kernel:
		gaussian_map = gen_gaussian_map(input_gt, gaussian_kernel, sigma)
	else:
		gaussian_map = fixed_gaussian_map(input_gt, gaussian_kernel, sigma)
	return gaussian_map


def fixed_gaussian_map(input_gt, kernel_size, sigma):
	kernel = (int(kernel_size), int(kernel_size))
	input_gt = np.array(input_gt, np.float)
	density_map = cv2.GaussianBlur(input_gt, kernel, sigma)
	density_map = density_map.astype(np.float32)
	return density_map


def gen_gaussian_map(input_gt, gaussian_kel, sig):
	h, w = input_gt.shape[:2]
	density_map = np.zeros_like(input_gt, dtype=np.float32)
	num_gt = input_gt.sum()
	if num_gt == 0:
		return density_map
	
	if num_gt < 20:
		return fixed_gaussian_map(input_gt, gaussian_kel, sig)
	
	point_key = get_point_key(input_gt)
	
	leafsize = 2048
	tree = scipy.spatial.KDTree(point_key.copy(), leafsize=leafsize)
	distances = tree.query(point_key, k=4)[0]
	
	for i in range(len(point_key)):
		point_x, point_y = min(h - 1, point_key[i][0]), min(w - 1, point_key[i][1])
		sigma = int(np.sum(distances[i][1:4]) // 3 * 0.3)
		sigma = max(1, sigma)
		sigma = min(sigma, 2)
		gaussian_radius = sigma * 3
		gaussian_map = np.multiply(
			cv2.getGaussianKernel(gaussian_radius * 2 + 1, sigma),
			cv2.getGaussianKernel(gaussian_radius * 2 + 1, sigma).T
		)
		x_left, x_right, y_up, y_down = 0, gaussian_map.shape[1], 0, gaussian_map.shape[0]
		if point_x < 0 or point_y < 0:
			continue
		if point_y < gaussian_radius:
			x_left = gaussian_radius - point_y
		if point_x < gaussian_radius:
			y_up = gaussian_radius - point_x
		if point_y + gaussian_radius >= w:
			x_right = gaussian_map.shape[1] - (gaussian_radius + point_y - w) - 1
		if point_x + gaussian_radius >= h:
			y_down = gaussian_map.shape[0] - (gaussian_radius + point_x - h) - 1
		
		density_map[
		max(0, point_x - gaussian_radius):min(h, point_x + gaussian_radius + 1),
		max(0, point_y - gaussian_radius):min(w, point_y + gaussian_radius + 1)
		] += gaussian_map[y_up:y_down, x_left:x_right]
	density_map = np.array(density_map, dtype=np.float32)
	return density_map


def get_point_key(input_gt):
	h, w = input_gt.shape[:2]
	gt_key = []
	for i in range(h):
		for j in range(w):
			if input_gt[i, j] == 1:
				gt_key.append([i, j])
	return gt_key
