import random

import torch
from torch.autograd import Variable
from torchvision import transforms


def to_var(tensor):
    """Converts tensor to variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor)


def to_tensor(variable):
    """Converts variable to tensor."""
    if torch.cuda.is_available():
        variable = variable.cpu()
    return variable.data


def to_image(tensor):
    """Converts tensor to pil Image."""
    return transforms.ToPILImage()(tensor.clamp(0, 1))


def summary(net):
    num_params = 0
    print("NAME | SIZE | NUM PARAMETERS")
    for name, param in net.named_parameters():
        print("{} | {} | {} ".format(name, tuple(param.size()), param.numel()))
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(model_state, optimizer_state, filename):
    state = dict(model_state=model_state,
                 optimizer_state=optimizer_state)
    torch.save(state, filename)
