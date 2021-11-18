import numpy as np
import torch
from tqdm import tqdm
import os
from utility.distributed import apply_gradient_allreduce, reduce_tensor
import torch.nn as nn
from copy import deepcopy
import torch.distributed as dist
from copy import deepcopy
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Utility
def select_dataset(option, addon_list, tr_dataset, tr_transform, val_transform, rank):
    addon_list[0].eval()
    batch_size, pin_memory = option.result['train']['batch_size'], option.result['train']['pin_memory']
    tr_dataset.dataset.transform = val_transform
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=option.result['train']['num_workers'])
    
    index_list = []
    for iter, data in enumerate(tr_loader):
        input, label = data
        input, label = input.to(rank), label.to(rank)
        
        with torch.no_grad():
            output, _ = addon_list[0](input, iter, rank, train=True, save=True)
        index = torch.argmax(output, dim=1).cpu().detach().numpy() != label.cpu().detach().numpy()
        index_list.append(index)
        
    index_list = np.concatenate(index_list)
    tr_dataset.select_dataset(index_list)
    tr_dataset.dataset.transform = tr_transform
    return tr_dataset



def accuracy(output, target, topk=(1,)):
    topk = (1,)
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def forward_single(option, input, label, model_list, addon_list, iter, rank, criterion_list, train=True, epoch=0, multi_gpu=False):
    train_method = option.result['train']['train_method']
        
    if epoch == 0:
        save = True
    else:
        save = False
        
    if train_method == 'base' or train_method == 'auxiliary':
        output = model_list[0](input)
        loss_cls = criterion_list[0](output, label)
        loss_channel = torch.tensor(0.).to(rank)
        
    elif train_method == 'selection':
        output, output_ori_list = model_list[0](input, iter, rank, train, save)
        
        loss_part = 0.
        for out in output_ori_list:
            loss_part += criterion_list[0](out, label)
        loss_part /= len(output_ori_list)

        # Total Loss
        loss_cls = (criterion_list[0](output, label) * 0.7 + loss_part * 0.3)
        loss_channel = torch.tensor(0.).to(rank)

    elif train_method == 'merge':
        output, output_ori_list = model_list[0](input, iter, rank, train, save)
        
        loss_part = 0.
        for out in output_ori_list:
            loss_part += criterion_list[0](out, label)
        loss_part /= len(output_ori_list)

        # Total Loss
        loss_cls = (criterion_list[0](output, label) * 0.7 + loss_part * 0.3)
        loss_channel = torch.tensor(0.).to(rank)

    else:
        raise('Select Proper Train Methods')    

    return output, loss_cls, loss_channel


def train(option, rank, epoch, model_list, addon_list, criterion_list, optimizer_list, multi_gpu, tr_loader, scaler, save_module, neptune, save_folder):
    # GPU setup
    num_gpu = len(option.result['train']['gpu'].split(','))

    # For Log
    mean_loss_cls = 0.
    mean_loss_channel = 0.
    mean_acc1 = 0.

    # Freeze !
    train_method = option.result['train']['train_method']
    
    if train_method == 'base':
        model_list[0].train()
    elif train_method == 'selection':
        model_list[0].train()
    elif train_method == 'merge':
        model_list[0].train()
    elif train_method == 'auxiliary':
        model_list[0].train()
    else:
        raise('Select Proper Train Method')
    
    # Run
    for iter, tr_data in enumerate(tqdm(tr_loader)):
        input, label = tr_data
        input, label = input.to(rank), label.to(rank)

        # Forward
        optimizer_list[0].zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output, loss_cls, loss_channel = forward_single(option, input, label, model_list, addon_list, iter, rank, criterion_list, train=True, epoch=epoch, multi_gpu=multi_gpu)
                scaler.scale(loss_cls + loss_channel).backward()
                scaler.step(optimizer_list[0])
                scaler.update()
                
        else:
            output, loss_cls, loss_channel = forward_single(option, input, label, model_list, addon_list, iter, rank, criterion_list, train=True, epoch=epoch, multi_gpu=multi_gpu)
            (loss_cls + loss_channel).backward()
            optimizer_list[0].step()
            
        # Empty Un-necessary Memory
        torch.cuda.empty_cache()
        
        # Metrics        
        acc_result = accuracy(output, label, topk=(1, 5))

        if (num_gpu > 1) and (option.result['train']['ddp']):
            mean_loss_cls += reduce_tensor(loss_cls.data, num_gpu).item()
            mean_loss_channel += reduce_tensor(loss_channel.data, num_gpu).item()
            mean_acc1 += reduce_tensor(acc_result[0], num_gpu)

        else:
            mean_loss_cls += loss_cls.item()
            mean_loss_channel += loss_channel.item()
            mean_acc1 += acc_result[0]
        
        del output, loss_cls, loss_channel

    # Train Result
    mean_acc1 /= len(tr_loader)
    mean_loss_cls /= len(tr_loader)
    mean_loss_channel /= len(tr_loader)

    # Attention Value
    if train_method != 'base' and train_method != 'auxiliary':
        if multi_gpu:
            attn_selection = model_list[0].module.attn_selection.get_attention()
        else:
            attn_selection = model_list[0].attn_selection.get_attention()
        
    # Saving Network Params
    if option.result['tune']['tuning']:
        model_param = [None]
    else:
        model_param = []
        
        if multi_gpu:
            for model in model_list:
                model_param.append(deepcopy(model.module.state_dict()))
        else:
            for model in model_list:
                model_param.append(deepcopy(model.state_dict()))

    # Save
    save_module.save_dict['model'] = model_param
    save_module.save_dict['optimizer'] = [optimizer.state_dict() for optimizer in optimizer_list]
    save_module.save_dict['save_epoch'] = epoch

    if (rank == 0) or (rank == 'cuda'):
        # Logging
        print('Epoch-(%d/%d) - tr_ACC@1: %.2f, tr_loss_cls:%.3f, tr_loss_channel:%.3f' %(epoch, option.result['train']['total_epoch'], \
                                                                            mean_acc1, mean_loss_cls, mean_loss_channel))
        neptune['result/tr_loss_cls'].log(mean_loss_cls)
        neptune['result/tr_loss_channel'].log(mean_loss_channel)
        neptune['result/tr_acc1'].log(mean_acc1)
        
        if train_method != 'base' and train_method != 'auxiliary':
            neptune['result/attn_selection'].log(attn_selection)
        
        if not option.result['tune']['tuning']:
            # Save
            if epoch % option.result['train']['save_epoch'] == 0:
                torch.save({'model':model_param}, os.path.join(save_folder, 'epoch%d_model.pt' %epoch))

    if multi_gpu and (option.result['train']['ddp']) and not option.result['tune']['tuning']:
        dist.barrier()

    return save_module


def validation(option, rank, epoch, model_list, addon_list, criterion_list, multi_gpu, val_loader, scaler, neptune):
    # GPU
    num_gpu = len(option.result['train']['gpu'].split(','))
        
    # Freeze !
    train_method = option.result['train']['train_method']
    if train_method == 'base':
        model_list[0].eval()
    elif train_method == 'selection':
        model_list[0].eval()
    elif train_method == 'merge':
        model_list[0].eval()
    elif train_method == 'auxiliary':
        model_list[0].eval()
    else:
        raise('Select Proper Train Method')
    
    # For Log
    mean_loss_cls = 0.
    mean_loss_channel = 0.
    mean_acc1 = 0.

    with torch.no_grad():
        for iter, val_data in enumerate(tqdm(val_loader)):                
            input, label = val_data
            input, label = input.to(rank), label.to(rank)

            output, loss_cls, loss_channel = forward_single(option, input, label, model_list, addon_list, iter, rank, criterion_list, train=False, epoch=epoch, multi_gpu=multi_gpu)

            acc_result = accuracy(output, label, topk=(1, 5))

            if (num_gpu > 1) and (option.result['train']['ddp']):
                mean_loss_cls += reduce_tensor(loss_cls.data, num_gpu).item()
                mean_loss_channel += reduce_tensor(loss_channel.data, num_gpu).item()
                mean_acc1 += reduce_tensor(acc_result[0], num_gpu)

            else:
                mean_loss_cls += loss_cls.item()
                mean_loss_channel += loss_channel.item()
                mean_acc1 += acc_result[0]

    # Remove Un-neccessary Memory
    del output, loss_cls, loss_channel
    torch.cuda.empty_cache()
    
    # Train Result
    mean_acc1 /= len(val_loader)
    mean_loss_cls /= len(val_loader)
    mean_loss_channel /= len(val_loader)

    # Logging
    if (rank == 0) or (rank == 'cuda'):
        print('Epoch-(%d/%d) - val_ACC@1: %.2f, val_loss_cls:%.3f, val_loss_channel:%.3f' % (epoch, option.result['train']['total_epoch'], \
                                                                                mean_acc1, mean_loss_cls, mean_loss_channel))
        neptune['result/val_loss_cls'].log(mean_loss_cls)
        neptune['result/val_loss_channel'].log(mean_loss_channel)
        neptune['result/val_acc1'].log(mean_acc1)
        neptune['result/epoch'].log(epoch)

    result = {'acc1':mean_acc1, 'val_loss':mean_loss_cls + mean_loss_channel}

    if option.result['tune']['tuning']:
        tune.report(loss=(mean_loss_cls + mean_loss_channel), accuracy=mean_acc1)

    return result

def test():
    pass