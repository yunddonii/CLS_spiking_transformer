import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter, summary
from torchvision import transforms
import torchvision
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision
# from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy as TotalAcc
# from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score

# snn
from spikingjelly.clock_driven import functional

# timm
from timm.optim.optim_factory import create_optimizer_v2
from timm.models import create_model, load_checkpoint
from timm.utils import *
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data.auto_augment import augment_and_mix_transform, auto_augment_transform
from timm.data.random_erasing import RandomErasing
from timm.loss import LabelSmoothingCrossEntropy


from typing import Optional, Callable, Tuple
import os

from v_utils import *
from v_config import *
from v_dataloader import create_loader, load_data
import v_model
from v_model import TemporalBlock

from v_test import test, evaluation

torch.jit.fuser("fuser0")
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
torch.autograd.set_detect_anomaly(True)

def train_one_epoch(model, data_loader:DataLoader, criterion, optimizer, ratio:float, device:torch.device):
    
    model.train()
    model.train_mode = 'training'

    epoch_mse = 0
    epoch_ce = 0

    for data, label in data_loader:
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        
        if isinstance(output, tuple):
            # out, org_x, rec_x, consol_loss = output
            out, org_x, rec_x = output
            mse = ((org_x - rec_x)**2).mean()
            # mse = torch.norm((org_x - rec_x), p=1)
            # mse = ((org_x - rec_x)**2).mean() + F.kl_div(torch.log_softmax(rec_x), org_x, reduction="batchmean", log_target=True)
        else:
            out = output
            mse = torch.tensor([0]).to(device)
            
        ce = criterion(out, label)
        
        
        if isinstance(optimizer, tuple):
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            ce.backward(retain_graph=True)
            mse.backward(retain_graph=True)
            # (ce + mse * alpha).backward()
            optimizer[0].step()
            optimizer[1].step()
            
            epoch_mse += mse.item()
            epoch_ce += ce.item()
            
        else:
            optimizer.zero_grad()
            (ce + mse * ratio).backward(retain_graph=True)
            optimizer.step()
            epoch_ce += ce.item()
            if isinstance(output, tuple): epoch_mse += mse.item()

        functional.reset_net(model)

    if epoch_mse > 0:
        return {
            'loss' : epoch_ce/len(data_loader), 
            'mse' : epoch_mse/len(data_loader)
        }

    else:
        return {
            'loss' : epoch_ce/len(data_loader), 
            'mse' : 0.
        }


def val_one_epoch(model, data_loader:DataLoader, criterion, num_classes:int, device:torch.device):
    
    model.eval()
    # model.train_mode = 'testing'
    epoch_ce = 0
    epoch_mse = 0
    total_acc = 0 

    # tot_acc = MulticlassAccuracy().to(device)
    tot_acc = TotalAcc(num_classes=num_classes).to(device)
    acc = MulticlassAccuracy(num_classes=num_classes, average=None).to(device)
    f1 = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
    pre = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)
            
            # emb = temporal_model(data)
    
            output = model(data)
        
            if isinstance(output, tuple):
                # out, org_x, rec_x, consol_loss = output
                out, org_x, rec_x = output
                mse = ((org_x - rec_x)**2).mean()
                # mse = torch.norm((org_x - rec_x), p=1)
                
            else:
                out = output
                
            ce = criterion(out, label)

            epoch_ce += ce.item()
            if isinstance(output, tuple): epoch_mse += mse.item()

            functional.reset_net(model)

            pred = get_pred(out)
            
            # tot_acc.update(pred, label)
            tot_acc.update(pred, label)
            acc.update(pred, label)
            f1.update(pred, label)
            pre.update(pred, label)
            
    return {
        'loss' : epoch_ce/len(data_loader), 
        'mse' : epoch_mse/len(data_loader) if epoch_mse > 0 else 0., 
        'tot_acc' : tot_acc.compute(),
        # 'tot_acc' : total_acc/len(data_loader),
        'acc' : acc.compute(), 
        'f1' : f1.compute(), 
        'pre' : pre.compute(),
            }
        

def train(args:Config):
    
    train_writer = SummaryWriter(log_dir=args.save_log_path + '/train')
    val_writer = SummaryWriter(log_dir=args.save_log_path + '/val')

    # time series
    # loader, class_num_samples = create_loader(train=True, batch_size=args.batch_size, num_classes=args.num_classes, data_root=args.train_data_root, train_val_ratio=None, data_window_size=args.window_size, sampling=args.sampling, num_workers=args.num_workers)

    # if isinstance(loader, tuple):
    #     train_loader = loader[0]
    #     val_loader = loader[1]
        
    # else:
    #     train_loader = loader
    #     val_loader, _ = create_loader(train=False, batch_size=args.batch_size, num_classes=args.num_classes, data_root=args.test_data_root, data_window_size=args.window_size, num_workers=args.num_workers)

    # image
    train_transform = transforms.Compose([
        RandomResizedCropAndInterpolation(32, scale=(1.0, 1.0), ratio=(1.0, 1.0), interpolation='bicubic'),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.0),
        transforms.ColorJitter(0),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        RandomErasing(0.25, mode='const', max_area=1, device='cpu'),
        ])
    
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])


    train_set = torchvision.datasets.CIFAR10(root='../data', download=True, train=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root='../data', download=True, train=False, transform=test_transform)
    
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # dvs
    # dataset_train, dataset_test, train_sampler, test_sampler = load_data('../data', args.time_steps)
    
    # train_loader = torch.utils.data.DataLoader(
    #     dataset=dataset_train,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     drop_last=True)

    # val_loader = torch.utils.data.DataLoader(
    #     dataset=dataset_test,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     drop_last=False,)
    
    spikformer = create_model(
        'spikformer',
        pretrained=False,
        pretrained_cfg=None,
        pretrained_cfg_overlay=None,
        drop_rate=0.,
        drop_path_rate=0.,
        drop_block_rate=None,
        gating=args.gating,
        train_mode='training',
        img_h=32,img_w=32,patch_h=8,patch_w=8,
        # img_h=128, img_w=128, patch_h=32, patch_w=32,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        qkv_bias=False, 
        mlp_ratios=args.mlp_emb,
        depths=args.num_layers, 
        sr_ratios=1,
        time_num_layers=args.time_num_layers,
        T=args.time_steps, 
        lif_bias=args.bias, 
        data_patching_stride=args.stride,
        padding_patches=None,
        tau=args.tau,
        spk_encoding=args.spk_encoding,
        attn=args.attn, 
    )
    
    n_params = sum(p.numel() for p in spikformer.parameters() if p.requires_grad)
    print(f"creating model >> number of parameters : {n_params}")
    setattr(args, "model_params", n_params)
    
    spikformer = spikformer.to(args.device)
    functional.reset_net(spikformer)
    # time_block = time_block.to(args.device)
    
    # cal_weights = get_class_weights(class_num_samples=class_num_samples)

    train_criterion = LabelSmoothingCrossEntropy(0.1).to(args.device)
    val_criterion = nn.CrossEntropyLoss().to(args.device)

    optimizer1 = create_optimizer_v2(spikformer.parameters(), opt='adamw', lr=args.lr, weight_decay=args.weight_decay)

    scheduler1 = get_scheduler(args.scheduler, optimizer1, max_lr=args.max_lr, max_epochs=args.epoch)
    if args.scheduler == 'cosine' : scheduler1.step(0)

    patience = 0
    best_valid_loss = float("inf")
        
    for epoch in range(args.epoch):
        
        mse_ratio = args.alpha
        # if epoch < 10:
        # else:
        #     mse_ratio = 1e-6
        
        set_random_seed(42)
        
        train_loss = train_one_epoch(spikformer, train_loader, train_criterion, optimizer1, mse_ratio, args.device) 
        val_result = val_one_epoch(spikformer, val_loader, val_criterion, args.num_classes, args.device) 
        
        train_loss['tot_loss'] = train_loss['loss'] + train_loss['mse'] * args.alpha
        val_result['tot_val_loss'] = (val_result['loss'] + val_result['mse'] * args.alpha)
        
        if args.scheduler == 'reduce':
            scheduler1.step(val_result['tot_val_loss'])
            # scheduler2.step(val_result['mse'])
        else:
            scheduler1.step(epoch+1)
            # scheduler2.step(val_result['mse'])
        
        train_writer.add_scalar('train/loss', train_loss['tot_loss'], epoch)
        val_writer.add_scalar('val/loss', val_result['tot_val_loss'], epoch)
        val_writer.add_scalar('val/acc', val_result['acc'].clone().mean(), epoch)
        val_writer.add_scalar('val/f1', val_result['f1'].clone(), epoch)
        # summary_writer.add_pr_curve(f'val/pr_curve', val_result['pr_curve'],)
        
        if ((epoch + 1) % args.print_epoch) == 0:
            
            print_epoch_info(epoch=epoch,
                             train_loss=train_loss,
                             lr=optimizer1.param_groups[0]['lr'],
                             val_result=val_result,
                             num_classes=args.num_classes)
            
        if args.best_save :
            if val_result['tot_val_loss'] < best_valid_loss:
                patience+=1
                
        if (patience >= args.save_log_patience) or ((epoch + 1) == args.epoch):
            best_valid_loss = val_result['tot_val_loss']
            best_model = spikformer.state_dict() # inference mode
            
            torch.save(best_model, args.save_model_state_path + f"/{epoch:03d}+model.pt")
            print(f"Model saved to `{args.save_model_state_path}`")
            
            with open(args.save_log_path + f'/best+log.csv', 'a', encoding='utf-8') as log_csv:
                
                acc_class_values = ', '.join(f"{val_result['acc'][i]:.6f}" for i in range(args.num_classes))
                
                print(f"{epoch}", 
                        f"{train_loss['loss']:.6f}", 
                        f"{train_loss['mse']:.6f}", 
                        f"{val_result['loss']:.6f}", 
                        f"{val_result['mse']:.6f}", 
                        f"{val_result['tot_acc']:.6f}", 
                        f"{val_result['acc'].mean():.6f}", 
                        f"{val_result['f1']:.6f}", 
                        f"{val_result['pre']:.6f}", 
                        f"{acc_class_values}", 
                        sep=", ", end="\n", file=log_csv)
                
            print(f"Current log saved to `{args.save_log_path}`")
            
            patience = 0 if epoch < (args.epoch // 10) * 3 else 1
            args.saved_epoch.append(epoch)
                    
    
                    
    # final arguments save
    args.save_arg()
                
    train_writer.flush()
    train_writer.close()
    val_writer.flush()
    val_writer.close()
    
    if args.test:
        
        last_epoch = args.epoch - 1
        last_saved_epoch = args.saved_epoch[-1]
        
        if last_saved_epoch == last_epoch:
            evaluation(args=args, model=spikformer, loader=val_loader, criterion=val_criterion)
            
        else:
            evaluation(args=args, loader=val_loader, criterion=val_criterion)
        
if __name__ == '__main__' : 
    
    set_random_seed(42)
    
    config = parse_arguments()
    args = Config()
    
    args.set_args(config)
    args.print_info()
    
    with open(args.save_log_path + f'/best+log.csv', 'w', encoding='utf-8') as log_csv:
        
        acc_class_col = ', '.join(f"val_acc_{i}" for i in range(args.num_classes))
        print("epoch", "train_loss", "train_mse", "val_loss", "val_mse", "val_tot_acc", "val_acc", "val_f1", "val_pre", acc_class_col, sep=", ", end="\n", file=log_csv)

    train(args)