import torch
import torch.nn.functional as F

from utils.loss_fun import SSIM
from utils.ssim_loss import SSIM_Loss



def eval_net(model, data, factor, arg):
	model.eval()
	val_loss = 0
	step = 0
	if arg.loss_function == 'L2':
		criterion = torch.nn.MSELoss()
	if arg.loss_function == 'SmoothL1':
		criterion = torch.nn.SmoothL1Loss()
	if arg.loss_function == 'SSIM':
		criterion = SSIM()
	if arg.loss_function == 'SSIM_L2':
		criterion = SSIM()
	with torch.no_grad():
		for x, y in data:
			step = step + 1
			inputs = x.cuda()
			labels = y.cuda()
			if arg.model == 'TreeCountNet' or 'TEDnet':
				if arg.deepsupervision:
					out1, out2, out3, outputs = model(inputs)
					if arg.loss_function == 'SSIM':
						los1 = criterion(outputs, labels)
						los2 = criterion(outputs, labels)
						los3 = criterion(outputs, labels)
						loss_final = criterion(outputs, labels)
						# loss = 0.1 * los1 + 0.3 * los2 + 0.6 * los3 + loss_final
						loss = los1 + los2 + los3 + loss_final
					elif arg.loss_function == 'SSIM_L2':
						los1 = criterion(outputs, labels) * arg.ssim_weight + torch.nn.MSELoss()(outputs, labels)
						los2 = criterion(outputs, labels) * arg.ssim_weight + torch.nn.MSELoss()(outputs, labels)
						los3 = criterion(outputs, labels) * arg.ssim_weight + torch.nn.MSELoss()(outputs, labels)
						loss_final = criterion(outputs, labels) * arg.ssim_weight + torch.nn.MSELoss()(outputs, labels)
						# loss = 0.1 * los1 + 0.3 * los2 + 0.6 * los3 + loss_final
						loss = los1 + los2 + los3 + loss_final
					else:
						los1 = criterion(out1, labels)
						los2 = criterion(out2, labels)
						los3 = criterion(out3, labels)
						loss_final = criterion(outputs, labels)
						# loss = 0.1 * los1 + 0.3* los2 + 0.6 * los3 + loss_final
						loss = los1 + los2 + los3 + loss_final
				
				else:
					outputs = model(inputs)
					if arg.loss_function == 'SSIM':
						loss = criterion(outputs, labels)
					elif arg.loss_function == 'SSIM_L2':
						loss = criterion(outputs, labels) * arg.ssim_weight + torch.nn.MSELoss()(outputs, labels)
					else:
						loss = criterion(outputs, labels)
			else:
				outputs = model(inputs)
				if arg.loss_function == 'SSIM':
					loss = criterion(outputs, labels)
				elif arg.loss_function == 'SSIM_L2':
					ssim_loss = criterion(outputs, labels)
					l2_loss = torch.nn.MSELoss()(outputs, labels)
					loss = l2_loss + ssim_loss * arg.ssim_weight
				else:
					loss = criterion(outputs, labels)
			
			# loss = torch.nn.SmoothL1Loss()(pre, y)
			val_loss += float(loss)
			
	return val_loss / step
