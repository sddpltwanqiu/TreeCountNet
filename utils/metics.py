import torch


def mAE(SR, GT, factor):
	batch = SR.shape[0]
	error = 0
	for i in range(batch):
		sr = SR[i, 0, :, :]
		gt = GT[i, 0, :, :]
		sr[sr < 0] = 0
		preNum = sr.sum() / factor
		realNum = gt.sum() / factor
		error += torch.sub(preNum, realNum).abs()
	return error / batch


def R2(SR, GT, factor):
	batch = SR.shape[0]
	gt1 = torch.zeros([batch])
	pre = torch.zeros([batch])
	for i in range(batch):
		sr = SR[i, 0, :, :]
		gt = GT[i, 0, :, :]
		sr[sr < 0] = 0
		preNum = sr.sum() / factor
		realNum = gt.sum() / factor
		gt1[i] = realNum
		pre[i] = preNum
	x_mean = gt1.mean()
	y_mean = pre.mean()
	Top = 0.0
	x_pow = 0.0
	y_pow = 0.0
	for i in range(batch):
		Top += (gt1[i] - x_mean) * (pre[i] - y_mean)
		x_pow += (gt1[i] - x_mean) * (gt1[i] - x_mean)
		y_pow += (pre[i] - y_mean) * (pre[i] - y_mean)
	l = (x_pow * y_pow).sqrt()
	r2 = Top / l
	return r2


def RMSE(SR, GT, factor):
	batch = SR.shape[0]
	error = torch.zeros([batch])
	for i in range(batch):
		sr = SR[i, 0, :, :]
		gt = GT[i, 0, :, :]
		sr[sr < 0] = 0
		preNum = sr.sum() / factor
		realNum = gt.sum() / factor
		error[i] = torch.sub(preNum, realNum)
	rmse = error.std()
	return rmse


def sum1(GT, PRE, factor):
	batch = PRE.shape[0]
	for i in range(batch):
		gt_sum = 0
		pre_sum = 0
		pre = PRE[i, 0, :, :]
		gt = GT[i, 0, :, :]
		pre[pre < 0] = 0
		gt = gt / factor
		pre = pre / factor
		preNum = pre.sum()
		gtNum = gt.sum()
		gt_sum += gtNum
		pre_sum += preNum
	return gt_sum, pre_sum
