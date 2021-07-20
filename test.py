import argparse
import os
import numpy as np
import cv2
from time import *
import torch
from utils.metics import RMSE, mAE, R2
import torch.nn as nn
from torchvision.transforms import transforms

from networks import TreeCountNet
from utils.excel_tool import *
from networks import TEDNet
from utils.gaussian import choose_map_method
from utils.loss_fun import SSIM

from utils.metics import sum1


arg = parser.parse_args()

x_transforms = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()
out_num = []
out_ssim =[]
T = []
def test(model,device,dataset_path,checkpointpath,out_path,arg):
	model_name = 'weights_best_val_loss.pth'
	pt_name = checkpointpath + '/' + model_name
	if not os.path.exists(pt_name):
		continue
	else:
		book_name_xls = result_path + '/' + model_name[:-4] + '_pre_result.xls'

        test_img = dataset_path+'/IMG'
        test_gt = dataset_path+'/GT'
		
		if arg.model == 'MCNN':
			model = MCNN.MCNN()
		if arg.model == 'Crowdnet':
			model = Comnet.com_net()
		if arg.model == 'TreeCountNet':
			model = TreeCountNet.TreeCountNet(arg)
		if arg.model == 'TEDNet':
			model = TEDNet.TEDNet(arg)
		
		model = model.to(device)
		model = nn.DataParallel(model)
		model.load_state_dict(torch.load(pt_name))
		model.eval()
		
		name_list = os.listdir(test_img)
		name_l = []
		gt_l = []
		pre_l = []
		mae_1 = []
		ssim_1 = []
		for name in name_list:
			pth = os.path.join(test_img, name)
			pth2 = os.path.join(test_gt, name)
			gt = cv2.imread(pth2, flags=-1)
			gt = choose_map_method(gt, arg.gauss_kernel, arg.sigma, arg.adaption_gaussian)
			gt = gt * int(arg.factor)
			img = cv2.imread(pth, flags=-1)
			gt = y_transforms(gt)
			gt = torch.Tensor(gt)
			gt = gt.unsqueeze(0)
			img = x_transforms(img)
			img = img.unsqueeze(0)
			with torch.no_grad():
				x = img.to(device)
				y = gt.to(device)
				if arg.deepsupervision:
					_,_,_,pre1= model(x)
				else:
					pre1 = model(x)
				gt_sum, pre_sum = sum1(y, pre1, arg.factor)
				mae = mAE(y, pre1, arg.factor)
				ssim = SSIM()(y, pre1)
				mae_ = mae.cpu().numpy()
				gt_ = gt_sum.cpu().numpy()
				pre_ = pre_sum.cpu().numpy()
				ssim_ = ssim.cpu().numpy()
			out1 = pre1.cpu().numpy()
			out1 = np.squeeze(out1)
			outname = out_path+model_name+name
			cv2.imwrite(outname, out1)
			
			name_l.append(str(name))
			pre_l.append(str(pre_))
			gt_l.append(str(gt_))
			mae_1.append(str(mae_))
			ssim_1.append(str(ssim_))
