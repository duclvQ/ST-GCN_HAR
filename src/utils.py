import os
import shutil
import py3nvml
import torch

def check_gpu(gpus):
    if len(gpus) > 0 and torch.cuda.is_available():
        py3nvml.py3nvml.nvmlInit()
        for i in gpus:
            handle = py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = py3nvml.py3nvml.nvmlDeviceGetMemoryInfo(handle)
            memused = meminfo.used / 1024 / 1024
            print('GPU{} used: {}M'.format(i, memused))
            if memused > 1000:
                py3nvml.py3nvml.nvmlShutdown()
                raise ValueError('GPU{} is occupied!'.format(i))
        py3nvml.py3nvml.nvmlShutdown()
        return 'cuda'
    else:
        print('Using CPU!')
        return 'cpu'

def load_checkpoint(device_type, fname='checkpoint'):
    fpath = './models/' + fname + '.pth.tar'
    if os.path.isfile(fpath):
        checkpoint = torch.load(fpath, map_location=device_type)
        return checkpoint
    else:
        raise ValueError('Do NOT exist this checkpoint: {}'.format(fname))

def save_checkpoint(model, optimizer, epoch, best, is_best, model_name):
    if not os.path.exists('./models'):
        os.mkdir('./models')
    for key in model.keys():
        model[key] = model[key].cpu()
    checkpoint = {'model':model, 'optimizer':optimizer, 'epoch':epoch, 'best':best}
    torch.save(checkpoint, './models/checkpoint.pth.tar')
    if is_best:
        shutil.copy('./models/checkpoint.pth.tar', './models/' + model_name + '.pth.tar')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Cannot recognize the input parameter {}'.format(v))

