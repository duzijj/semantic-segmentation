import torch 
import argparse
import yaml
import time
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
# from torch.cuda.amp import GradScaler, autocast
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from semseg.models import *
from semseg.datasets import * 
# from semseg.augmentations import get_train_augmentation, get_val_augmentation
from semseg.ext_transforms import transform
from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp
# from mmseg.datasets import build_dataset, build_dataloader
from val import evaluate
import os


def main(cfg, gpu, save_dir, args):
    global model, img, lbl, trainset
    start = time.time()
    best_mIoU = 0.0
    num_workers = max(4,mp.cpu_count())
    # num_workers = 4
    
    # device = torch.device(cfg['DEVICE']) if torch.cuda.is_available() else 'cpu'
    device = args.device if torch.cuda.is_available() else 'cpu'
    print('DEVICE:'+ device)
    print(cfg['MODEL'])
    
    device = torch.device(device)
    
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']
    
    traintransform, valtransform = transform(train_cfg['IMAGE_SIZE'])

    
    
    # if cfg['DATASET']['NAME']=='CityScapes':
    #     trainset = build_dataset(train_config)
    #     valset = build_dataset(val_config)
    #     testset = build_dataset(test_config)
    # else:
    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train', traintransform)
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', valtransform)
    testset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'test', valtransform)
    
    
    
    
    
    if 'BiSeNet' in model_cfg['NAME']:
        # model = eval(model_cfg['NAME'])(trainset.n_classes)
        model = eval(model_cfg['NAME'])(trainset.n_classes, model_cfg['SPATIAL'], model_cfg['SEMANTIC'])
    elif 'DDRNet' in model_cfg['NAME']:
        model = eval(model_cfg['NAME'])(num_classes=trainset.n_classes, backbone=model_cfg['BACKBONE'],
                                        strategy1=model_cfg['STRATEGY1'], strategy2=model_cfg['STRATEGY2'],
                                        fuse=model_cfg['FUSE'], PRETRAINED=model_cfg['PRETRAINED'])
    # model.init_pretrained(model_cfg['PRETRAINED'])
    if train_cfg['DDP']: 
        model = DDP(model, device_ids=[0,1])
    model = model.to(device)
    
    
    
    trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True, pin_memory=False, shuffle=True)
    valloader = DataLoader(valset, batch_size=2, num_workers=num_workers, pin_memory=False, shuffle=True)
    testloader = DataLoader(testset, batch_size=2, num_workers=num_workers, pin_memory=False, shuffle=True)

    iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE']
    # class_weights = trainset.class_weights.to(device)
    loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, None)
    loss_fn = loss_fn.to(device)

        
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, epochs * iters_per_epoch, sched_cfg['POWER'], iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])
    # scaler = GradScaler(enabled=train_cfg['AMP'])
    
    
    print("=====> computing network parameters and FLOPs")
    total_paramters = netParams(model)
    print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))
    
    # logFileLoc = '%s/%s_%s_%s_%s.txt'%(save_dir, dataset_cfg['NAME'], model_cfg['NAME'], 
    #                                 model_cfg['BACKBONE'] if model_cfg['BACKBONE'] else model_cfg['SPATIAL'],
    #                                 model_cfg['SEMANTIC'] if 'SEMANTIC' in model_cfg.keys() else '')
    logFileLoc = '%s/%s.txt'%(save_dir, args.name)
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        
    logger.write('\n\n' + time.strftime('%Y-%m-%d[%H:%M]') + " the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))
    logger.flush()
    
    
    for epoch in range(epochs):
        model.train()
        # if train_cfg['DDP']: sampler.set_epoch(epoch)
        training_time_start = time.time()
        train_loss = 0.0
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")

        for iter, (img, lbl) in pbar:
            optimizer.zero_grad()
            
            img = img.to(device)
            lbl = lbl.long().to(device)
            
            # if train_cfg['DDP']: 
            #     img = DDP(img)
            #     lbl = DDP(lbl)
            
            logits = model(img)
            loss = loss_fn(logits, lbl)

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            loss.backward()
            optimizer.step()
            # scaler.update()
            scheduler.step()
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()

            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            train_loss += loss.item()

            pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter+1):.8f}")
        
        train_loss /= iter+1
        # torch.cuda.empty_cache()

        if (epoch+1) % train_cfg['EVAL_INTERVAL'] == 0 or (epoch+1) == epochs:
            model.eval()
            if cfg['DATASET']['NAME']=='CamVid':
                miou = evaluate(model, testloader, device)[-1]
            else:
                miou = evaluate(model, valloader, device)[-1]
            logger.write('\n' +'Epoch:%d/%d'%(epoch+1, epochs) + ' | ' + 'miou:' + str(miou) + ' | loss:%.2f'%train_loss + ' | lr:%2f'%lr)
            logger.flush()


            if miou > best_mIoU:
                best_mIoU = miou
                # torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}.pth")
                torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), save_dir / (args.name+'.pth'))
            
            time_taken_epoch = time.time() - training_time_start
            remain_time = time_taken_epoch * (epochs - 1 - epoch)
            m, s = divmod(remain_time, 60)
            h, m = divmod(m, 60)
            print(f"Current mIoU: {miou} Best mIoU: {best_mIoU}  Remaining Time: %d:%d:%d"%(h, m, s))
            
            # if ((epoch+1) % 100 == 0 or (epoch+1) == epochs) and :
            #     miou = evaluate(model, testloader, device)[-1]
            #     print('Test mIou:' + str(miou))
            #     logger.write('\n' +'Test mIou:' + str(miou))
            #     logger.flush()
            
            ## trick
            # if miou < best_mIoU-0.03:
            #     model.load_state_dict(torch.load(str(save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}.pth"), map_location='cpu'))
            #     model = model.to(device)
            
    # miou = evaluate(model, testloader, device)[-1]
    # print('Test mIou:' + str(miou))
    logger.write('\n' +'Best mIou:' + str(best_mIoU))
    logger.flush()
    logger.close()
    pbar.close()
    end = time.gmtime(time.time() - start)

    table = [
        ['Best mIoU', f"{best_mIoU:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    print(tabulate(table, numalign='right'))
    print("=====> computing network parameters and FLOPs")
    total_paramters = netParams(model)
    print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))

def netParams(model):
    """
    computing total network parameters
    args:
       model: model
    return: the number of parameters
    """
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='camvid_ddrnet_se_BFUSE', help='Configuration file to use')
    parser.add_argument('--dataset', type=str, default='camvid', help='Configuration file to use')
    parser.add_argument('--device', type=str, default='cuda', help='Configuration file to use')
    args = parser.parse_args()
    args.cfg = '../configs/%s/%s.yaml'%(args.dataset, args.name)
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp()
    save_dir = Path(cfg['SAVE_DIR'] + '/%s'%args.dataset)
    save_dir.mkdir(exist_ok=True)
    
    
    main(cfg, gpu, save_dir, args)
    cleanup_ddp()