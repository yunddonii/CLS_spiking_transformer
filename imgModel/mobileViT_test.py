from v_utils import *
from v_config import *
from v_dataloader import create_loader, load_data
import v_model
from mobileViT import MobileViT
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
from v_test import test, get_model_complexity_info

from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision
# from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy as TotalAcc
# from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score

def evaluation(args:Config, criterion):
    
    model = MobileViT().to(args.device)
    saved_model_path = os.path.join(args.save_result_path, "model_state", f"{47:03d}+model.pt")
    model.load_state_dict(torch.load(saved_model_path))
    # loader, class_num_samples = create_loader(train=True, batch_size=args.batch_size, num_classes=args.num_classes, data_root=args.train_data_root, train_val_ratio=args.train_val_ratio, data_window_size=args.window_size, sampling=args.sampling, num_workers=args.num_workers)
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    # train_set = torchvision.datasets.CIFAR10(root='../data', download=True, train=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root='../data', download=True, train=False, transform=test_transform)
    
    # train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    overall_acc = MulticlassAccuracy(num_classes=args.num_classes).to(args.device)
    acc = MulticlassAccuracy(num_classes=args.num_classes, average=None).to(args.device)
    f1 = MulticlassF1Score(num_classes=args.num_classes, average='macro').to(args.device)
    pre = MulticlassPrecision(num_classes=args.num_classes, average='macro').to(args.device)

    model.eval()

    
    with torch.no_grad():
        epoch_loss = 0
        sim = 0
        for data, label in val_loader:
            data = data.to(args.device)
            label = label.to(args.device)

            output = model(data)

            loss = criterion(output, label)
            epoch_loss += loss.item()

            pred = get_pred(output)
            
            overall_acc.update(pred, label)
            acc.update(pred, label)
            f1.update(pred, label)
            pre.update(pred, label)
            
        input_res = data[0].shape
        
        model_info_per_layer_path = os.path.join(args.save_log_path, 'model+info+per+layer.txt') 
        file_out = open(model_info_per_layer_path, 'w', encoding='utf-8')
       
        # model.train_mode = 'testing'
        ops, params, fr = get_model_complexity_info(
                                    model=model,
                                    input_res=(input_res,), 
                                    dataloader=val_loader,
                                    as_strings=False,
                                    print_per_layer_stat=True,
                                    # custom_modules_hooks=modules,
                                    # ignore_modules=ignore_modules,
                                    verbose=False,
                                    ost=file_out,
                                )
        
        file_out.close()
        
    test_result = {
        'loss' : epoch_loss/len(val_loader),
        'overall_acc' : overall_acc.compute(),
        'acc' : acc.compute(),
        'f1' : f1.compute(),
        'pre' : pre.compute(),
        # 'sim' : sim/len(loader)
    }
    
    print("Test was successfully done")

    with open(args.save_log_path + '/final+result.csv', 'w', encoding='utf-8') as log_csv:
        acc_col = ', '.join(f"acc{i}" for i in range(args.num_classes))
        print("loss", "overall_acc", "average_acc", "f1", "precision", "sensitivity", "total_op", "ACop", "MACop", "capacity", "firing_rate", "energy", acc_col,
              sep=", ", end="\n", file=log_csv)
        
        acc_value = ', '.join(f"{test_result['acc'][i]:.6f}" for i in range(args.num_classes))
            
        print(f"{test_result['loss']:.6f}", 
              f"{test_result['overall_acc']:.6f}",
              f"{test_result['acc'].clone().mean():.6f}", 
              f"{test_result['f1']:.6f}", 
              f"{test_result['pre']:.6f}", 
            #   f"{test_result['sim']:.6f}",
              f"{ops[0] / 1e6:.2f} M Ops",
              f"{ops[1] / 1e6:.2f} M Ops",
              f"{ops[2] / 1e6:.2f} M Ops",
              f"{params / 1e6:.4f} M",
              f"{fr:.4f} %",
              f"{get_energy_consumption(O_ac=ops[1], O_mac=ops[2], unit='u'):.2f} uJ",
              acc_value,
              sep=", ", end="\n", file=log_csv)
        
    print(f"Final result saved to `{args.save_log_path}`")
    
if __name__ == '__main__':
    
    set_random_seed(42)
    
    config = parse_arguments()
    args = Config()
    
    if config.config:
        config_path = config.config
        args.load_args(config_path, config)
    
    else:
        args.set_args(config)
        
    args.print_info()
    val_criterion = nn.CrossEntropyLoss().to(args.device)

    evaluation(args, criterion=val_criterion)