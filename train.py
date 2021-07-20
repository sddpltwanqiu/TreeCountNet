import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from utils.dataset import make_Dataset
from utils.eval import eval_net
from utils.ssim_loss import SSIM_Loss


def train_model(model, criterion, optimizer, dataload, valid_dataload, device, checkpoint_path, result_path, num_epochs,
                factor, arg):
    best_loss = None
    best_val_loss = None
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 30)
        epoch_loss = 0
        mse1 = 0
        mae1 = 0
        r2_1 = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device).float()

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            if arg.model == 'TreeCountNet' or 'TENet':
                if arg.deepsupervision:
                    out1, out2, out3, outputs = model(inputs)

                    if arg.loss_function == 'SSIM':
                        los1 = criterion(outputs, labels)
                        los2 = criterion(outputs, labels)
                        los3 = criterion(outputs, labels)
                        loss_final = criterion(outputs, labels)
                        #loss = 0.1 * los1 + 0.3 * los2 + 0.6 * los3 + loss_final
                        loss = los1 + los2 + los3 + loss_final
                    elif arg.loss_function == 'SSIM_L2':
                        los1 = criterion(
                            outputs, labels) * arg.ssim_weight + torch.nn.MSELoss()(outputs, labels)
                        los2 = criterion(
                            outputs, labels) * arg.ssim_weight + torch.nn.MSELoss()(outputs, labels)
                        los3 = criterion(
                            outputs, labels) * arg.ssim_weight + torch.nn.MSELoss()(outputs, labels)
                        loss_final = criterion(
                            outputs, labels) * arg.ssim_weight + torch.nn.MSELoss()(outputs, labels)
                        loss = los1 + los2 + los3 + loss_final
                    else:
                        los1 = criterion(out1, labels)
                        los2 = criterion(out2, labels)
                        los3 = criterion(out3, labels)
                        loss_final = criterion(outputs, labels)
                        loss = los1 + los2 + los3 + loss_final

                else:
                    outputs = model(inputs)

                    if arg.loss_function == 'SSIM':
                        loss = criterion(outputs, labels)
                    elif arg.loss_function == 'SSIM_L2':
                        loss = criterion(
                            outputs, labels) * arg.ssim_weight + torch.nn.MSELoss()(outputs, labels)
                    else:
                        loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)

                if arg.loss_function == 'SSIM':
                    loss = criterion(outputs, labels)
                elif arg.loss_function == 'SSIM_L2':
                    ssim_loss = criterion(outputs, labels)
                    l2_loss = torch.nn.MSELoss()(outputs, labels)
                    loss = l2_loss + ssim_loss*arg.ssim_weight
                else:
                    loss = criterion(outputs, labels)

            epoch_loss += np.float(loss)
            loss.backward()
            optimizer.step()

        train_loss = epoch_loss / step

        print("epoch %d loss:%0.6f mse:%0.6f, mae:%0.6f, r2:%0.6f" % (
            epoch, epoch_loss / step, mse1 / step, mae1 / step, r2_1 / step))
        val_loss, val_mAE, val_mse, val_r2 = eval_net(
            model, valid_dataload, factor, arg)
        print('Valid_Loss: {},Valid_MAE: {}, Valid_MSE:{},Valid_R2:{}'.format(
            val_loss, val_mAE, val_mse, val_r2))

        if best_loss == None:
            best_loss = train_loss
            torch.save(model.state_dict(), checkpoint_path +
                       'weights_best_loss.pth')
            print('Checkpoint 0 saved !'.format(epoch))
        elif train_loss < best_loss:
            torch.save(model.state_dict(), checkpoint_path +
                       'weights_best_loss.pth')
            best_loss = train_loss
            print('Checkpoint {} train_loss improved and saved !'.format(epoch))
        elif epoch == num_epochs - 1:
            torch.save(model.state_dict(),
                       checkpoint_path + 'weights_last.pth')
            print('Checkpoint {} improved and saved !'.format(epoch))
        else:
            print('Best_loss not improved !')

        if best_val_loss == None:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path +
                       'weights_best_val_loss.pth')
        elif val_loss < best_val_loss:
            torch.save(model.state_dict(), checkpoint_path +
                       'weights_best_val_loss.pth')
            best_val_loss = val_loss
            print('Checkpoint {} val_loss improved and saved !'.format(epoch))
        else:
            print('Best_val_loss not improved !')

    return model


# 训练模型
def train(model, device, data_path, checkpoint_path, result_path, arg):
    model = model.to(device)
    model = nn.DataParallel(model)
    batch_size = arg.batch_size
    if arg.loss_function == 'L2':
        criterion = torch.nn.MSELoss()
    if arg.loss_function == 'SSIM':
        criterion = SSIM_Loss()
    if arg.loss_function == 'SSIM_L2':
        criterion = SSIM_Loss()

    # optimizer = optim.Adam(model.parameters(),weight_decay=10.0)
    optimizer = optim.Adam(model.parameters())
    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    y_transforms = transforms.ToTensor()
    dataset_path = data_path
    train_path = dataset_path + '/train'
    train_dataset = make_Dataset(
        root=train_path, arg=arg, transform=x_transforms, target_transform=y_transforms)
    len_data = int(train_dataset.__len__())
    train, valid = torch.utils.data.random_split(
        train_dataset, [1728, len_data - 1728])
    # valid_dataset = make_Dataset(
    #    valid, arg=arg, transform=x_transforms, target_transform=y_transforms)
    train_dataloader = DataLoader(
        train, batch_size=batch_size, num_workers=4)
    valid_dataloader = DataLoader(
        valid, batch_size=batch_size, num_workers=4)
    train_model(model, criterion, optimizer, train_dataloader, valid_dataloader, device=device,
                checkpoint_path=checkpoint_path, result_path=result_path, num_epochs=arg.num_epochs, factor=arg.factor,
                arg=arg)
