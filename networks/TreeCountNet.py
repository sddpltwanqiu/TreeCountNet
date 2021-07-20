import torch
import argparse
import torch.nn as nn


def ConvBNReLU(in_channels, out_channels, kernel_size):
	return nn.Sequential(
		nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
		          padding=kernel_size // 2),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True)
	)


class FME(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(FME, self).__init__()
		
		self.branch1_conv = ConvBNReLU(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=1)
		
		self.branch2_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=1)
		self.branch2_conv2 = ConvBNReLU(in_channels=out_channels // 2, out_channels=out_channels // 4, kernel_size=3)
		
		self.branch3_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=1)
		self.branch3_conv2 = ConvBNReLU(in_channels=out_channels // 2, out_channels=out_channels // 4, kernel_size=5)
		
		self.branch4_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=1)
		self.branch4_conv2 = ConvBNReLU(in_channels=out_channels // 2, out_channels=out_channels // 4, kernel_size=7)
	
	def forward(self, x):
		out1 = self.branch1_conv(x)
		out2 = self.branch2_conv2(self.branch2_conv1(x))
		out3 = self.branch3_conv2(self.branch3_conv1(x))
		out4 = self.branch4_conv2(self.branch4_conv1(x))
		out = torch.cat([out1, out2, out3, out4], dim=1)
		return out


class FME_first(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(FME_first, self).__init__()
		
		self.branch1_conv = ConvBNReLU(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=1)
		self.branch2_conv = ConvBNReLU(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=3)
		self.branch3_conv = ConvBNReLU(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=5)
		self.branch4_conv = ConvBNReLU(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=7)
	
	def forward(self, x):
		out1 = self.branch1_conv(x)
		out2 = self.branch2_conv(x)
		out3 = self.branch3_conv(x)
		out4 = self.branch4_conv(x)
		out = torch.cat([out1, out2, out3, out4], dim=1)
		return out


class TreeCountNet(nn.Module):
	def __init__(self, args):
		super(TreeCountNet, self).__init__()
		
		self.args = args
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		
		self.conv0_0 = FME_first(in_channels=3, out_channels=32)
		self.conv1_0 = FME(in_channels=32, out_channels=64)
		self.up1_0 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
		self.conv0_1 = FME(in_channels=64, out_channels=32)
		
		self.conv2_0 = FME(in_channels=64, out_channels=128)
		self.up1_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
		self.conv1_1 = FME(in_channels=160, out_channels=64)
		self.up0_2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
		self.conv0_2 = FME(in_channels=96, out_channels=32)
		
		self.conv3_0 = FME(in_channels=128, out_channels=256)
		self.up2_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
		self.conv2_1 = FME(in_channels=320, out_channels=128)
		self.up1_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
		self.conv1_2 = FME(in_channels=224, out_channels=64)
		self.up0_3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
		self.conv0_3 = FME(in_channels=128, out_channels=32)
		
		self.conv4_0 = FME(in_channels=256, out_channels=512)
		self.up3_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
		self.conv3_1 = FME(in_channels=640, out_channels=256)
		self.up2_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
		self.conv2_2 = FME(in_channels=448, out_channels=128)
		self.up1_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
		self.conv1_3 = FME(in_channels=288, out_channels=64)
		self.up0_4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
		self.conv0_4 = FME(in_channels=160, out_channels=32)
		
		if self.args.deepsupervision:
			self.final1 = nn.Conv2d(32, 1, kernel_size=1)
			self.final2 = nn.Conv2d(32, 1, kernel_size=1)
			self.final3 = nn.Conv2d(32, 1, kernel_size=1)
			self.final4 = nn.Conv2d(32, 1, kernel_size=1)
		else:
			self.final = nn.Conv2d(32, 1, kernel_size=1)
	
	def forward(self, x):
		out0_0 = self.conv0_0(x)
		out1_0 = self.conv1_0(self.pool(out0_0))
		out0_1 = self.conv0_1(torch.cat([out0_0, self.up1_0(out1_0)], 1))
		
		out2_0 = self.conv2_0(self.pool(out1_0))
		out1_1 = self.conv1_1(torch.cat([self.pool(out0_1),out1_0, self.up1_1(out2_0)], 1))
		out0_2 = self.conv0_2(torch.cat([out0_0, out0_1, self.up0_2(out1_1)], 1))
		
		out3_0 = self.conv3_0(self.pool(out2_0))
		out2_1 = self.conv2_1(torch.cat([self.pool(out1_1),out2_0, self.up2_1(out3_0)], 1))
		out1_2 = self.conv1_2(torch.cat([self.pool(out0_2),out1_0, out1_1, self.up1_2(out2_1)], 1))
		out0_3 = self.conv0_3(torch.cat([out0_0, out0_1, out0_2, self.up0_3(out1_2)], 1))
		
		out4_0 = self.conv4_0(self.pool(out3_0))
		out3_1 = self.conv3_1(torch.cat([self.pool(out2_1), out3_0, self.up3_1(out4_0)], 1))
		out2_2 = self.conv2_2(torch.cat([self.pool(out1_2), out2_0, out2_1, self.up2_2(out3_1)], 1))
		out1_3 = self.conv1_3(torch.cat([self.pool(out0_3), out1_0, out1_1, out1_2, self.up1_3(out2_2)], 1))
		out0_4 = self.conv0_4(torch.cat([out0_0, out0_1, out0_2, out0_3, self.up0_4(out1_3)], 1))
		
		if self.args.deepsupervision:
			output1 = self.final1(out0_1)
			output2 = self.final2(out0_2)
			output3 = self.final3(out0_3)
			output4 = self.final4(out0_4)
			return output1, output2, output3, output4
		
		else:
			output = self.final(out0_4)
			return output

def print_model_parm_nums(model):
	total = sum([param.nelement() for param in model.parameters()])
	return total

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--deepsupervision', type=bool, default=True)
	arg = parser.parse_args()
	A = torch.randn(2, 3, 256, 256)
	model = TreeCountNet(args=arg)
	total = print_model_parm_nums(model)
	out = model(A)
	print('  + Number of params: %.2fM' % (total / 1e6))
	#print(out.shape)
