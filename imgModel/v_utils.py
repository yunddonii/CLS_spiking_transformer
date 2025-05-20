import torch
import pandas as pd
    
import math
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn as nn
from timm.scheduler import create_scheduler, create_scheduler_v2

from copy import deepcopy
from spikingjelly.clock_driven import functional

# visualization
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from tqdm import tqdm
import os

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    



def undersampling(data, sampling_size):

    num_classes = len(data.iloc[:, -1].unique())
    
    if sampling_size:
        min_value = sampling_size
    else:
        min_value = data.iloc[:, -1].value_counts().min()
    
    X_sampled = []
    y_sampled = []

    start_idx = 0
    for i in range(0, num_classes):
        
        max_idx = len(data[data.iloc[:, -1] == i])
        mask = np.random.permutation(range(start_idx, start_idx + max_idx))
        mask = mask[:min_value]
        sampled_data = data.iloc[mask, :]
        
        X_sampled.append(sampled_data.iloc[:, :-1].values)
        y_sampled.append(sampled_data.iloc[:, -1].values)
        
        start_idx = start_idx + max_idx
        
    df_X = np.vstack(X_sampled)
    df_y = np.hstack(y_sampled)

    df_X = pd.DataFrame(df_X)
    df_y = pd.DataFrame(df_y)
    
    df = pd.concat([df_X, df_y], axis=1, ignore_index=False)
    
    return df

    
def get_scheduler(scheduler:str, optimizer, **kargs):
    
    step_size = 10
    gamma = 0.5 
    T_max = 10
    patience = 2

    if scheduler == 'step' : 
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)

    elif scheduler == 'exponential' : scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # elif scheduler == 'cosine' : scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0)
    # elif scheduler == 'cosine' : scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=0)
    elif scheduler == 'cosine' : 
        max_lr = kargs["max_lr"]
        max_epochs = kargs["max_epochs"]
        # scheduler = create_scheduler()
        # scheduler = torch.optim.lr_scheduler.cos
        # scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=10, cycle_mult=1, min_lr=1e-10, max_lr=max_lr, gamma=0.9, warmup_steps=3)
        scheduler, _ = create_scheduler_v2(optimizer, sched='cosine', num_epochs=max_epochs, decay_epochs=30, warmup_epochs=20, cooldown_epochs=10, min_lr=0.002, noise_pct=0.67, warmup_lr=0.00001)
        # scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=max_epochs, cycle_mult=1, min_lr=1e-10, max_lr=max_lr, gamma=0.9, warmup_steps=2)
    elif scheduler == 'reduce' : scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.1, cooldown=1, min_lr=1e-6)
    else:
        assert NotImplementedError(scheduler)
    
    return scheduler

def get_pred(output, topk=(1,)):
    
    maxk = min(max(topk), output.shape[1])
    _, pred = output.topk(maxk, 1, True, True)
    
    return pred.squeeze(-1)
        
        
def get_class_weights(class_num_samples:torch.Tensor,) -> torch.Tensor :
    
    num_classes = class_num_samples.shape[0]
    
    ## Class-aware loss
    num_max = class_num_samples.max()
    tot_num_samples = class_num_samples.sum()
    
    mu = num_max / tot_num_samples
    
    if num_classes < 5:
        weights = mu * tot_num_samples / class_num_samples
    else:
        weights = torch.log(mu * tot_num_samples / class_num_samples)
        weights = torch.max(torch.ones_like(weights), weights)
    
    # weights = tot_num_samples / class_num_samples
    print(weights)
    
    ## Context-aware loss
    # total = np.sum(class_num_samples)
    # class_weights = dict()
    # num_classes = len(class_num_samples)
    
    # factor = 1 / num_classes
    # mu = [factor]
    
    return weights


def model_info(model, verbose=False, img_size=640):
    """ (C) copyright https://github.com/BICLab/EMS-YOLO

    Args:
        model (_type_): _description_
        verbose (bool, optional): _description_. Defaults to False.
        img_size (int, optional): _description_. Defaults to 640.
    """
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPs
        from thop import profile
        
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # input
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        fs = ', %.1f GFLOPs' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPs
    except (ImportError, Exception):
        fs = ''
        
    
def get_energy_consumption(O_ac, O_mac, E_ac=0.9, E_mac=4.6, unit=None) -> float:
    
    E_ac = E_ac * 1e-12
    E_mac = E_mac * 1e-12
    
    energy = E_ac * O_ac + E_mac * O_mac
    
    if unit is None or unit == 'p':
        return energy / 1e-12
    
    elif unit == 'm' :
        return energy / 1e-3
    
    elif unit == 'u' :
        return energy / 1e-6
    
def tsne_visual(feature:list, actual_label, metric, save_path, num_classes):

    tsne = TSNE(n_components=2, metric=metric, perplexity=20)
    cluster = np.array(tsne.fit_transform(np.array(feature)))
    actual = np.array(actual_label)

    plt.figure(figsize=(10, 10))
    labels = [str(i) for i in range(num_classes)]
    
    color = ["#66C5CC","#F6CF71","#F89C74","#DCB0F2","#87C55F","#9EB9F3","#FE88B1","#C9DB74","#8BE0A4","#B497E7","#D3B484","#B3B3B3"]
    
    for l, label in tqdm(enumerate(labels)):
        idx = np.where(actual == l)
        plt.scatter(cluster[idx, 0], cluster[idx, 1], marker='.', label=label, c=color[l])

    plt.savefig(save_path)
    plt.close()
    
    
def heatmap_visual(features:np.array, save_path):
    
    os.makedirs(save_path, exist_ok=True)
    for batch in range(len(features)):
        for sample in tqdm(range(features[batch].shape[1])):
            feature = features[batch].sum(0)[sample].sum(0)
            
            fig, ax = plt.subplots()
            heatmap = ax.pcolor(feature, cmap=plt.cm.Blues)
            
            ax.set_xticks(np.arange(feature.shape[1]) + 0.5, minor=False)
            ax.set_yticks(np.arange(feature.shape[0]) + 0.5, minor=False)
            
            ax.set_xlim(0, int(feature.shape[1]))
            ax.set_ylim(0, int(feature.shape[0]))
            
            ax.invert_yaxis()
            ax.xaxis.tick_top()
            
            ax.set_xlabel('key')
            ax.set_ylabel('query')
            
            plt.xticks(rotation=45)
            plt.savefig(save_path+f'/{batch}_{sample}_attn_map.png')
            plt.close()
    
        
def plot_eval(model, loader, num_classes, save_path, device):

    actual = []
    embedding = {'attn_map' : list(),}

    with torch.no_grad():
        for data, label in loader:
            
            data, label = data.to(device), label.to(device)
            
            feature = model(data)
            
            for i, emb_k in enumerate(feature):
                
                if not emb_k in embedding.keys():
                        embedding[emb_k] = feature[emb_k].cpu().numpy().tolist()
                else:
                    if emb_k == 'attn_map':
                        embedding[emb_k].append(feature[emb_k].cpu().data.numpy())
                    else:
                        embedding[emb_k] += feature[emb_k].cpu().numpy().tolist()
                    
                
            actual += label.cpu().numpy().tolist()
            
            functional.reset_net(model)

    
    for i, emb_k in enumerate(embedding):    
        if emb_k == 'attn_map' :
            # pass
            heatmap_visual(embedding[emb_k], save_path=save_path + f'/attn_heatmap_result')
        else:
            tsne_visual(embedding[emb_k], actual_label=actual, metric='euclidean', save_path=save_path + f'/tsne_result_{emb_k}.png', num_classes=num_classes)
        

def print_epoch_info(epoch, train_loss, lr, val_result:dict, num_classes:int):
    bar = "-" * 30
    
    print(bar)
    print(f"{'epoch':16s}{epoch:>15d} (lr={lr})")

    for k, v in train_loss.items():
        value = v if isinstance(v, float) else v.mean()
        print(f"{'train_' + k:16s}{value:>15.5f}")

    print(bar)
        
    for k, v in val_result.items():
        value = v if isinstance(v, float) else v.mean()
        print(f"{'val_' + k:16s}{value:>15.5f}")
    print(bar)
            
    # for i in range(num_classes): print(f"{'val_' + 'acc' + '_' + str(i):15s}{val_result['acc'][i]:>15.5f}")
    # print(bar)
    
    
# class OrthogonalProjectionLoss(nn.Module):
#     def __init__(self, no_norm=False, use_attention=False, gamma=2):
#         super(OrthogonalProjectionLoss, self).__init__()

#         self.no_norm = no_norm
#         self.gamma = gamma
#         self.use_attention = use_attention


#     def forward(self, features, labels=None):
        
#         device = features.device

#         if self.use_attention:
#             features_weights = torch.matmul(features, features.T)
#             features_weights = F.softmax(features_weights, dim=1)
#             features = torch.matmul(features_weights, features)

#         #  features are normalized
#         if not self.no_norm:
#             features = F.normalize(features, p=2, dim=1)

#         labels = labels[:, None]  # extend dim
#         mask = torch.eq(labels, labels.t()).bool().to(device)
#         eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

#         mask_pos = mask.masked_fill(eye, 0).float()
#         mask_neg = (~mask).float()
#         dot_prod = torch.matmul(features, features.t())

#         pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
#         neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

#         loss = (1.0 - pos_pairs_mean) + (self.gamma * neg_pairs_mean)
#         # loss = neg_pairs_mean

#         return loss, pos_pairs_mean, neg_pairs_mean