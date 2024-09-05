"""
A generic training script that works with any model and dataset.
"""

import argparse
from pathlib import Path
import signal
import shutil
import re
import os
import copy
from collections import defaultdict
import numpy as np

import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from omegaconf import open_dict
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from datasets import get_dataset
from models import get_model
from utils.stdout_capturing import capture_outputs
from pixloc.pixlib.utils.tools import AverageMetric, MedianMetric, set_seed, fork_rng
from pixloc.pixlib.utils.tensor import batch_to_device
from pixloc.pixlib.utils.experiments import (delete_old_checkpoints, get_last_checkpoint, get_best_checkpoint)
from pixloc.settings import TRAINING_PATH
from pixloc import logger

from pixloc.pixlib.utils.wandb_logger import WandbLogger
from pixloc.pixlib.datasets.kitti_voc import corruptions
from pixloc.pixlib.utils.experiments import load_experiment
import datetime

default_train_conf = {
    'seed': 20,  # training seed
    'epochs': 1,  # number of epochs
    'optimizer': 'adam',  # name of optimizer in [adam, sgd, rmsprop]
    'opt_regexp': None,  # regular expression to filter parameters to optimize
    'optimizer_options': {},  # optional arguments passed to the optimizer
    'lr': 0.0001,  # learning rate
    'lr_schedule': {'type': None, 'start': 0, 'exp_div_10': 0, 'gamma': 0.5}, #
    'lr_scaling': [(100, ['dampingnet.const'])],
    'eval_every_iter': 1000,  # interval for evaluation on the validation set
    'log_every_iter': 200,  # interval for logging the loss to the console
    'keep_last_checkpoints': 10,  # keep only the last X checkpoints
    'load_experiment': None,  # initialize the model from a previous experiment
    'median_metrics': [],  # add the median of some metrics
    'best_key': 'loss/total',  # key to use to select the best checkpoint
    'dataset_callback_fn': None,  # data func called at the start of each epoch
    'clip_grad': None,
}
default_train_conf = OmegaConf.create(default_train_conf)

def do_evaluation(model, loader, device, loss_fn, metrics_fn, conf, pbar=True, wandb_logger=None):
    model.eval()
    results = {}
    acc = 0
    total = 0
    errR = torch.tensor([])
    errlong = torch.tensor([])
    errlat = torch.tensor([])
    err_topk = 0

    for data in tqdm(loader, desc='Evaluation', ascii=True, disable=not pbar):
        data = batch_to_device(data, device, non_blocking=True)
        with torch.no_grad():
            pred = model(data)
            losses = loss_fn(pred, data)
            metrics = metrics_fn(pred, data)

            errR = torch.cat([errR, metrics['R_error'].cpu().data], dim=0)
            errlong = torch.cat([errlong, metrics['long_error'].cpu().data], dim=0)
            errlat = torch.cat([errlat, metrics['lat_error'].cpu().data], dim=0)
            if 'topk_err' in pred.keys():
                err_topk += pred['topk_err'].cpu().data

            del pred, data
        numbers = {**metrics, **{'loss/'+k: v for k, v in losses.items()}}
        for k, v in numbers.items():
            if k not in results:
                results[k] = AverageMetric()
                if k in conf.median_metrics:
                    results[k+'_median'] = MedianMetric()
            results[k].update(v)
            if k in conf.median_metrics:
                results[k+'_median'].update(v)
    results = {k: results[k].compute() for k in results}

    # if lat <= 0.2 and long <= 0.4 and R < 1: #requerment of Ford
    logger.info(f'acc of lat<=0.25:{torch.sum(errlat <= 0.25) / errlat.size(0)}')
    logger.info(f'acc of lat<=0.5:{torch.sum(errlat <= 0.5) / errlat.size(0)}')
    logger.info(f'acc of lat<=1:{torch.sum(errlat <= 1) / errlat.size(0)}')
    logger.info(f'acc of lat<=2:{torch.sum(errlat <= 2) / errlat.size(0)}')

    logger.info(f'acc of long<=0.25:{torch.sum(errlong <= 0.25) / errlong.size(0)}')
    logger.info(f'acc of long<=0.5:{torch.sum(errlong <= 0.5) / errlong.size(0)}')
    logger.info(f'acc of long<=1:{torch.sum(errlong <= 1) / errlong.size(0)}')
    logger.info(f'acc of long<=2:{torch.sum(errlong <= 2) / errlong.size(0)}')

    # logger.info(f'acc of R<=0.5:{torch.sum(errR <= 0.5) / errR.size(0)}')
    logger.info(f'acc of R<=1:{torch.sum(errR <= 1) / errR.size(0)}')
    logger.info(f'acc of R<=2:{torch.sum(errR <= 2) / errR.size(0)}')
    logger.info(f'acc of R<=4:{torch.sum(errR <= 4) / errR.size(0)}')

    logger.info(f'mean errR:{torch.mean(errR)},errlat:{torch.mean(errlat)},errlong:{torch.mean(errlong)}')
    logger.info(f'var errR:{torch.var(errR)},errlat:{torch.var(errlat)},errlong:{torch.var(errlong)}')
    logger.info(f'median errR:{torch.median(errR)},errlat:{torch.median(errlat)},errlong:{torch.median(errlong)}')

    if wandb_logger != None:
        wandb_logger.wandb.log({'val/lat 1m': torch.sum(errlat <= 1) / errlat.size(0)})
        wandb_logger.wandb.log({'val/lon 1m': torch.sum(errlong <= 1) / errlong.size(0)})
        wandb_logger.wandb.log({'val/rot 1': torch.sum(errR <= 1) / errR.size(0)})

        if isinstance(err_topk, torch.Tensor):
            wandb_logger.wandg.log({'val/T_1 err': err_topk[0] /  errlat.size(0)})
            wandb_logger.wandg.log({'val/T_topk err': err_topk[1] / errlat.size(0)})
            wandb_logger.wandg.log({'val/R_1 err': err_topk[2] / errlat.size(0)})
            wandb_logger.wandg.log({'val/R_topk err': err_topk[3] / errlat.size(0)})

    return results


def test_basic(dataset, model, wandb_logger=None):
    test_loader = dataset.get_data_loader('test', shuffle=False)

    model.eval()
    results = {}
    errR = torch.tensor([])
    errlong = torch.tensor([])
    errlat = torch.tensor([])
    err_topk = 0

    for idx, data in enumerate(tqdm(test_loader)):
        data_ = batch_to_device(data, device='cuda')
        # logger.set(data_)
        pred_ = model(data_)
        metrics = model.metrics(pred_, data_)

        errR = torch.cat([errR, metrics['R_error'].cpu().data], dim=0)
        errlong = torch.cat([errlong, metrics['long_error'].cpu().data], dim=0)
        errlat = torch.cat([errlat, metrics['lat_error'].cpu().data], dim=0)
        if 'topk_err' in pred.keys():
            err_topk += pred['topk_err'].cpu().data

        del pred_, data_

    #     for k, v in metrics.items():
    #         if k not in results:
    #             results[k] = AverageMetric()
    #             if k in conf.median_metrics:
    #                 results[k + '_median'] = MedianMetric()
    #         results[k].update(v)
    #         if k in conf.median_metrics:
    #             results[k + '_median'].update(v)
    # results = {k: results[k].compute() for k in results}

    logger.info(f'acc of lat<=0.25:{torch.sum(errlat <= 0.25) / errlat.size(0)}')
    logger.info(f'acc of lat<=0.5:{torch.sum(errlat <= 0.5) / errlat.size(0)}')
    logger.info(f'acc of lat<=1:{torch.sum(errlat <= 1) / errlat.size(0)}')
    logger.info(f'acc of lat<=2:{torch.sum(errlat <= 2) / errlat.size(0)}')

    logger.info(f'acc of long<=0.25:{torch.sum(errlong <= 0.25) / errlong.size(0)}')
    logger.info(f'acc of long<=0.5:{torch.sum(errlong <= 0.5) / errlong.size(0)}')
    logger.info(f'acc of long<=1:{torch.sum(errlong <= 1) / errlong.size(0)}')
    logger.info(f'acc of long<=2:{torch.sum(errlong <= 2) / errlong.size(0)}')

    logger.info(f'acc of R<=1:{torch.sum(errR <= 1) / errR.size(0)}')
    logger.info(f'acc of R<=2:{torch.sum(errR <= 2) / errR.size(0)}')
    logger.info(f'acc of R<=4:{torch.sum(errR <= 4) / errR.size(0)}')

    logger.info(f'mean errR:{torch.mean(errR)}, errlat:{torch.mean(errlat)}, errlong:{torch.mean(errlong)}')
    logger.info(f'var errR:{torch.var(errR)}, errlat:{torch.var(errlat)}, errlong:{torch.var(errlong)}')
    logger.info(f'median errR:{torch.median(errR)}, errlat:{torch.median(errlat)}, errlong:{torch.median(errlong)}')

    if wandb_logger != None:
        wandb_logger.wandb.log({'test/lat 0.25m': torch.sum(errlat <= 0.25) / errlat.size(0)})
        wandb_logger.wandb.log({'test/lat 0.5m': torch.sum(errlat <= 0.5) / errlat.size(0)})
        wandb_logger.wandb.log({'test/lat 1m': torch.sum(errlat <= 1) / errlat.size(0)})
        wandb_logger.wandb.log({'test/mean errlat': torch.mean(errlat)})
        wandb_logger.wandb.log({'test/var errlat': torch.var(errlat)})
        wandb_logger.wandb.log({'test/median errlat': torch.median(errlat)})

        wandb_logger.wandb.log({'test/lon 0.25m': torch.sum(errlong <= 0.25) / errlong.size(0)})
        wandb_logger.wandb.log({'test/lon 0.5m': torch.sum(errlong <= 0.5) / errlong.size(0)})
        wandb_logger.wandb.log({'test/lon 1m': torch.sum(errlong <= 1) / errlong.size(0)})
        wandb_logger.wandb.log({'test/mean errlon': torch.mean(errlong)})
        wandb_logger.wandb.log({'test/var errlon': torch.var(errlong)})
        wandb_logger.wandb.log({'test/median errlon': torch.median(errlong)})

        wandb_logger.wandb.log({'test/rot 1': torch.sum(errR <= 1) / errR.size(0)})
        wandb_logger.wandb.log({'test/rot 2': torch.sum(errR <= 2) / errR.size(0)})
        wandb_logger.wandb.log({'test/rot 4': torch.sum(errR <= 4) / errR.size(0)})
        wandb_logger.wandb.log({'test/mean errR': torch.mean(errR)})
        wandb_logger.wandb.log({'test/var errR': torch.var(errR)})
        wandb_logger.wandb.log({'test/median errR': torch.median(errR)})

        if isinstance(err_topk, torch.Tensor):
            wandb_logger.wandg.log({'val/T_1 err': err_topk[0] /  errlat.size(0)})
            wandb_logger.wandg.log({'val/T_topk err': err_topk[1] / errlat.size(0)})
            wandb_logger.wandg.log({'val/R_1 err': err_topk[2] / errlat.size(0)})
            wandb_logger.wandg.log({'val/R_topk err': err_topk[3] / errlat.size(0)})


    return

def test_kitti_voc(dataset, model, wandb_logger=None):
    # load dataloader
    test_loaders = {}
    for corruption in corruptions:
        test_loader = dataset.get_corruption_data_loader(corruption)
        test_loaders[corruption] = test_loader

    # test
    model.eval()
    total_err = {}
    np.random.seed(0)

    for corruption, test_loader in test_loaders.items():
        print(corruption)
        severity_err = {}

        for idx, data in enumerate(tqdm(test_loader)):
        # for idx, data in zip(range(3), test_loader):
            severity_images = data['query']['image'].copy()

            for severity, severity_image in severity_images.items():
                data['query']['image'] = severity_image # it use only a specific severity grd image

                data_ = batch_to_device(data, device='cuda')
                # logger.set(data_)
                pred_ = model(data_)
                metrics = model.metrics(pred_, data_)

                if severity not in severity_err:
                    severity_err[severity] = []
                severity_err[severity].append(torch.stack([metrics['R_error'].cpu().data, metrics['long_error'].cpu().data, metrics['lat_error'].cpu().data])) # (3, B)

                del pred_, data_

        for severity in severity_err.keys():
            severity_err[severity] = torch.concat(severity_err[severity], dim=-1).permute(1, 0) # (N x B, 3)
        total_err[corruption] = severity_err

    # print
    for corruption, severity_err in total_err.items():
        logger.info(f'[corruption: {corruption}]')

        errlat = []
        errlong = []
        errR = []
        for severity, err in severity_err.items():
            n = err.size(0)
            logger.info(f'- [{severity}] acc of lat<=1:{torch.sum(err[:, 2] <= 1) / n}')
            logger.info(f'- [{severity}] acc of long<=1:{torch.sum(err[:, 1] <= 1) / n}')
            logger.info(f'- [{severity}] acc of R<=1:{torch.sum(err[:, 0] <= 1) / n}')

            logger.info(f'- [{severity}] mean errR:{torch.mean(err[:, 0])}, errlat:{torch.mean(err[:, 2])}, errlong:{torch.mean(err[:, 1])}')
            logger.info(f'- [{severity}] var errR:{torch.var(err[:, 0])}, errlat:{torch.var(err[:, 2])}, errlong:{torch.var(err[:, 1])}')
            logger.info(f'- [{severity}] median errR:{torch.median(err[:, 0])}, errlat:{torch.median(err[:, 2])}, errlong:{torch.median(err[:, 1])}')

            errlat.append(err[:, 2])
            errlong.append(err[:, 1])
            errR.append(err[:, 0])

        errlat = torch.concat(errlat, dim=0)
        errlong = torch.concat(errlong, dim=0)
        errR = torch.concat(errR, dim=0)

        logger.info(f'- [total] acc of lat<=1:{torch.sum(errlat <= 1) / errlat.size(0)}')
        logger.info(f'- [total] acc of long<=1:{torch.sum(errlong <= 1) / errlong.size(0)}')
        logger.info(f'- [total] acc of R<=1:{torch.sum(errR <= 1) / errR.size(0)}')

        logger.info(f'- [total] mean errR:{torch.mean(errR)}, errlat:{torch.mean(errlat)}, errlong:{torch.mean(errlong)}')
        logger.info(f'- [total] var errR:{torch.var(errR)}, errlat:{torch.var(errlat)}, errlong:{torch.var(errlong)}')
        logger.info(f'- [total] median errR:{torch.median(errR)}, errlat:{torch.median(errlat)}, errlong:{torch.median(errlong)}')

        if wandb_logger != None:
            wandb_logger.wandb.log({f'total/rot 1': torch.sum(errR <= 1) / errR.size(0)})
            wandb_logger.wandb.log({f'total/lat 1': torch.sum(errlat <= 1) / errlat.size(0)})
            wandb_logger.wandb.log({f'total/lon 1': torch.sum(errlong <= 1) / errlong.size(0)})

            wandb_logger.wandb.log({f'total/mean errR': torch.mean(errR)})
            wandb_logger.wandb.log({f'total/mean errlat': torch.mean(errlat)})
            wandb_logger.wandb.log({f'total/mean errlong': torch.mean(errlong)})

            wandb_logger.wandb.log({f'total/var errR': torch.var(errR)})
            wandb_logger.wandb.log({f'total/var errlat': torch.var(errlat)})
            wandb_logger.wandb.log({f'total/var errlong': torch.var(errlong)})

            wandb_logger.wandb.log({f'total/median errR': torch.median(errR)})
            wandb_logger.wandb.log({f'total/median errlat': torch.median(errlat)})
            wandb_logger.wandb.log({f'total/median errlong': torch.median(errlong)})

            wandb_logger.wandb.log({f'{corruption}/lat 0.25m': torch.sum(errlat <= 0.25) / errlat.size(0)})
            wandb_logger.wandb.log({f'{corruption}/lat 0.5m': torch.sum(errlat <= 0.5) / errlat.size(0)})
            wandb_logger.wandb.log({f'{corruption}/lat 1m': torch.sum(errlat <= 1) / errlat.size(0)})
            wandb_logger.wandb.log({f'{corruption}/mean errlat': torch.mean(errlat)})
            wandb_logger.wandb.log({f'{corruption}/var errlat': torch.var(errlat)})
            wandb_logger.wandb.log({f'{corruption}/median errlat': torch.median(errlat)})

            wandb_logger.wandb.log({f'{corruption}/lon 0.25m': torch.sum(errlong <= 0.25) / errlong.size(0)})
            wandb_logger.wandb.log({f'{corruption}/lon 0.5m': torch.sum(errlong <= 0.5) / errlong.size(0)})
            wandb_logger.wandb.log({f'{corruption}/lon 1m': torch.sum(errlong <= 1) / errlong.size(0)})
            wandb_logger.wandb.log({f'{corruption}/mean errlon': torch.mean(errlong)})
            wandb_logger.wandb.log({f'{corruption}/var errlon': torch.var(errlong)})
            wandb_logger.wandb.log({f'{corruption}/median errlon': torch.median(errlong)})

            wandb_logger.wandb.log({f'{corruption}/rot 1': torch.sum(errR <= 1) / errR.size(0)})
            wandb_logger.wandb.log({f'{corruption}/rot 2': torch.sum(errR <= 2) / errR.size(0)})
            wandb_logger.wandb.log({f'{corruption}/rot 4': torch.sum(errR <= 4) / errR.size(0)})
            wandb_logger.wandb.log({f'{corruption}/mean errR': torch.mean(errR)})
            wandb_logger.wandb.log({f'{corruption}/var errR': torch.var(errR)})
            wandb_logger.wandb.log({f'{corruption}/median errR': torch.median(errR)})


    return

def test(rank, conf, output_dir, args, wandb_logger=None):
    if args.distributed:
        logger.info(f'Training in distributed mode with {args.n_gpus} GPUs')
        assert torch.cuda.is_available()
        #torch.distributed.init_process_group(
        #        backend='nccl', world_size=args.n_gpus, rank=rank,
        #        init_method= 'env://')

        # 1 gpu 1 progress
        device = rank
        # lock = Path(os.getcwd(),
        #             f'distributed_lock_{os.getenv("LSB_JOBID", device)}')
        # assert not Path(lock).exists(), lock
        torch.distributed.init_process_group(
                backend='nccl', world_size=args.n_gpus, rank=device, #'gloo'
                init_method= 'file://'+str(args.lock_file),timeout=datetime.timedelta(seconds=60))
        torch.cuda.set_device(device)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open_dict(conf):
        conf.model.optimizer.max_num_points3D = conf.data.max_num_points3D
    model = load_experiment(args.experiment, conf, ckpt=f'/ws/external/outputs/training/{args.experiment}/checkpoint_best.tar').to(device)

    data_conf = copy.deepcopy(conf.data)
    # load dataset
    dataset = get_dataset(data_conf.name)(data_conf)

    # test
    if data_conf.name == 'kitti_voc':
        test_kitti_voc(dataset, model, wandb_logger)
    else:
        test_basic(dataset, model, wandb_logger)

    return

def filter_parameters(params, regexp):
    '''Filter trainable parameters based on regular expressions.'''
    # Examples of regexp:
    #     '.*(weight|bias)$'
    #     'cnn\.(enc0|enc1).*bias'
    def filter_fn(x):
        n, p = x
        match = re.search(regexp, n)
        if not match:
            p.requires_grad = False
        return match
    params = list(filter(filter_fn, params))
    assert len(params) > 0, regexp
    logger.info('Selected parameters:\n'+'\n'.join(n for n, p in params))
    return params


def pack_lr_parameters(params, base_lr, lr_scaling):
    '''Pack each group of parameters with the respective scaled learning rate.
    '''
    filters, scales = tuple(zip(*[
        (n, s) for s, names in lr_scaling for n in names]))
    scale2params = defaultdict(list)
    for n, p in params:
        scale = 1
        # TODO: use proper regexp rather than just this inclusion check
        is_match = [f in n for f in filters]
        if any(is_match):
            scale = scales[is_match.index(True)]
        scale2params[scale].append((n, p))
    logger.info('Parameters with scaled learning rate:\n%s',
                {s: [n for n, _ in ps] for s, ps in scale2params.items()
                 if s != 1})
    lr_params = [{'lr': scale*base_lr, 'params': [p for _, p in ps]}
                 for scale, ps in scale2params.items()]
    return lr_params

def linear_annealing(init, fin, step, start_step=2000, end_step=6000):
    assert fin > init
    assert end_step > start_step
    if step < start_step:
        return init
    if step > end_step:
        return fin

    delta = fin - init
    annealed = min(init + delta * (step - start_step) / (end_step - start_step), fin)
    return annealed

def training(rank, conf, output_dir, args, wandb_logger=None):
    if args.restore:
        logger.info(f'Restoring from previous training of {args.experiment}')
        # init_cp = get_last_checkpoint(args.experiment, allow_interrupted=False)
        # logger.info(f'Restoring from checkpoint {init_cp.name}')
        init_cp = os.path.join(output_dir, 'checkpoint_last.tar')
        init_cp = torch.load(str(init_cp), map_location='cpu')
        conf = OmegaConf.merge(OmegaConf.create(init_cp['conf']), conf)
        epoch = init_cp['epoch'] + 1

        # get the best loss or eval metric from the previous best checkpoint
        best_cp = get_best_checkpoint(args.experiment)
        best_cp = torch.load(str(best_cp), map_location='cpu')
        best_eval = best_cp['eval'][conf.train.best_key]
        del best_cp

    # elif args.finetune:
    #     conf.train = OmegaConf.merge(default_train_conf, conf.train)
    #     epoch = 0
    #     best_eval = float('inf')
    #
    #     logger.info(f'Finetuning from {args.finetune}')
    #
    #     # get the best loss or eval metric from the previous best checkpoint
    #     init_cp = torch.load(args.finetune)
    #     best_eval = float('inf')

    else:
        # we start a new, fresh training
        conf.train = OmegaConf.merge(default_train_conf, conf.train)
        epoch = 0
        best_eval = float('inf')
        if conf.train.load_experiment:
            logger.info(
                f'Will fine-tune from weights of {conf.train.load_experiment}')
            # the user has to make sure that the weights are compatible
            # init_cp = get_last_checkpoint(conf.train.load_experiment)
            init_cp = os.path.join('/ws/external/outputs/training', conf.train.load_experiment, 'checkpoint_best.tar')
            init_cp = torch.load(str(init_cp), map_location='cpu')
        else:
            init_cp = None

    OmegaConf.set_struct(conf, True)  # prevent access to unknown entries
    set_seed(conf.train.seed)

    if rank == 0:
        if wandb_logger==None:
            writer = SummaryWriter(log_dir=str(output_dir))
            wandb_logger = WandbLogger(None)

    data_conf = copy.deepcopy(conf.data)
    if args.distributed:
        logger.info(f'Training in distributed mode with {args.n_gpus} GPUs')
        assert torch.cuda.is_available()
        #torch.distributed.init_process_group(
        #        backend='nccl', world_size=args.n_gpus, rank=rank,
        #        init_method= 'env://') 

        # 1 gpu 1 progress
        device = rank
        # lock = Path(os.getcwd(),
        #             f'distributed_lock_{os.getenv("LSB_JOBID", device)}')
        # assert not Path(lock).exists(), lock
        torch.distributed.init_process_group(
                backend='nccl', world_size=args.n_gpus, rank=device, #'gloo'
                init_method= 'file://'+str(args.lock_file),timeout=datetime.timedelta(seconds=60))
        torch.cuda.set_device(device)

        # adjust batch size and num of workers since these are per GPU
        if 'batch_size' in data_conf:
            data_conf.batch_size = int(data_conf.batch_size / args.n_gpus)
        if 'train_batch_size' in data_conf:
            data_conf.train_batch_size = int(
                data_conf.train_batch_size / args.n_gpus)
        if 'num_workers' in data_conf:
            data_conf.num_workers = int(
                (data_conf.num_workers + args.n_gpus - 1) / args.n_gpus)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device {device}')

    dataset = get_dataset(data_conf.name)(data_conf)
    if args.overfit:
        # we train and eval with the same single training batch
        logger.info('Data in overfitting mode')
        assert not args.distributed
        train_loader = dataset.get_overfit_loader('train')
        val_loader = dataset.get_overfit_loader('val')
    else:
        train_loader = dataset.get_data_loader(
            'train', distributed=args.distributed)
        val_loader = dataset.get_data_loader('val')
        test_loader = dataset.get_data_loader('test', shuffle=False)

    if rank == 0:
        logger.info(f'Training loader has {len(train_loader)} batches')
        logger.info(f'Validation loader has {len(val_loader)} batches')

    # interrupts are caught and delayed for graceful termination
    def sigint_handler(signal, frame):
        logger.info('Caught keyboard interrupt signal, will terminate')
        nonlocal stop
        if stop:
            raise KeyboardInterrupt
        stop = True
    stop = False
    signal.signal(signal.SIGINT, sigint_handler)

    OmegaConf.set_struct(conf, True)
    with open_dict(conf):
        conf.model.optimizer.max_num_points3D = conf.data.max_num_points3D

    model = get_model(conf.model.name)(conf.model).to(device)
    loss_fn, metrics_fn = model.loss, model.metrics
    if init_cp is not None:
        model.load_state_dict(init_cp['model'])

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device],
            static_graph=False,
            find_unused_parameters=True)
        # model._set_static_graph()
    if rank == 0:
        logger.info(f'Model: \n{model}')
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)

    optimizer_fn = {'sgd': torch.optim.SGD,
                    'adam': torch.optim.Adam,
                    'rmsprop': torch.optim.RMSprop}[conf.train.optimizer]
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if conf.train.opt_regexp:
        params = filter_parameters(params, conf.train.opt_regexp)
    all_params = [p for n, p in params]

    if 'lr_scaling' not in conf.train.keys():
        lr_params = pack_lr_parameters(
            params, conf.train.lr, default_train_conf.lr_scaling)
    else:
        lr_params = pack_lr_parameters(
                params, conf.train.lr, conf.train.lr_scaling)
    optimizer = optimizer_fn(
            lr_params, lr=conf.train.lr, **conf.train.optimizer_options)

    # lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, 1)

    # def lr_fn(it):  # noqa: E306
    #     if conf.train.lr_schedule.type is None:
    #         return 1
    #     elif conf.train.lr_schedule.type == 'exp':
    #         gam = 10**(-1/conf.train.lr_schedule.exp_div_10)
    #         return 1 if it < conf.train.lr_schedule.start else gam
    #     else:
    #         raise ValueError(conf.train.lr_schedule.type)
    # lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fn)
    if conf.train.lr_schedule.type is None:
        def lr_fn(it):
            return 1
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fn)
    elif conf.train.lr_schedule.type == 'exp':
        def lr_fn(it):
            gam = 10**(-1/conf.train.lr_schedule.exp_div_10)
            return 1 if it < conf.train.lr_schedule.start else gam
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fn)
    elif conf.train.lr_schedule.type == 'step':
        total_epoch = conf.train.epochs
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[total_epoch//2, total_epoch*3//4],
                                                            gamma=conf.train.lr_schedule.gamma)
    elif conf.train.lr_schedule.type == 'lambda':
        total_epoch = conf.train.epochs
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: ((1.0 - float(epoch) / total_epoch)**1.0))

    if args.restore:
        #optimizer.load_state_dict(init_cp['optimizer']) # delte because para not same after add satellite feature extractor
        if 'lr_scheduler' in init_cp:
            lr_scheduler.load_state_dict(init_cp['lr_scheduler'])

    if rank == 0:
        logger.info('Starting training with configuration:\n%s',
                    OmegaConf.to_yaml(conf))
    losses_ = None

    # torch.cuda.empty_cache()  # should be cleared at the first iter

    while epoch < conf.train.epochs and not stop:
        if rank == 0:
            logger.info(f'Starting epoch {epoch}')
        set_seed(conf.train.seed + epoch)
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        # if epoch > 0 and conf.train.dataset_callback_fn:
        #     getattr(train_loader.dataset, conf.train.dataset_callback_fn)(
        #         conf.train.seed + epoch)
        errR = torch.tensor([])
        errlong = torch.tensor([])
        errlat = torch.tensor([])
        err_topk = 0

        for it, data in enumerate(train_loader):
            tot_it = len(train_loader)*epoch + it
            model.train()
            optimizer.zero_grad()
            data = batch_to_device(data, device, non_blocking=True)
            pred = model(data)
            losses = loss_fn(pred, data)
            loss = torch.mean(losses['total'])
            metrics = metrics_fn(pred, data)

            with torch.no_grad():
                errR = torch.cat([errR, metrics['R_error'].cpu().data], dim=0)
                errlong = torch.cat([errlong, metrics['long_error'].cpu().data], dim=0)
                errlat = torch.cat([errlat, metrics['lat_error'].cpu().data], dim=0)
                if 'topk_err' in pred.keys():
                    err_topk += pred['topk_err'].cpu().data

            #tick = time.time()
            do_backward = loss.requires_grad
            if args.distributed:
                do_backward = torch.tensor(do_backward).float().to(device)
                torch.distributed.all_reduce(
                        do_backward, torch.distributed.ReduceOp.PRODUCT)
                do_backward = do_backward > 0
            if do_backward:
                loss.backward()
                #logger.info(f'after backward, time{time.time()-tick}')
                optimizer.step()
                # lr_scheduler.step()
                if conf.train.get('clip_grad', None):
                    if it % conf.train.log_every_iter == 0:
                        grads = [p.grad.data.abs().reshape(-1)
                                 for p in all_params if p.grad is not None]
                        ratio = (torch.cat(grads, 0) > conf.train.clip_grad)
                        ratio = ratio.float().mean().item()*100
                        if ratio > 25:
                            logger.warning(
                                f'More than {ratio:.1f}% of the parameters'
                                ' are larger than the clip value.')
                        del grads, ratio
                    torch.nn.utils.clip_grad_value_(
                            all_params, conf.train.clip_grad)
            else:
                if rank == 0:
                    logger.warning(f'Skip iteration {it} due to detach.')

            if it % conf.train.log_every_iter == 0:
                for k in sorted(losses.keys()):
                    if args.distributed:
                        losses[k] = losses[k].sum()
                        torch.distributed.reduce(losses[k], dst=0)
                        losses[k] /= (train_loader.batch_size * args.n_gpus)
                    losses[k] = torch.mean(losses[k]).item()
                if rank == 0:
                    str_losses = [f'{k} {v:.3E}' for k, v in losses.items()]
                    logger.info('[E {} | it {}] loss {{{}}}'.format(
                        epoch, it, ', '.join(str_losses)))

                    if args.wandb:
                        for k, v in losses.items():
                            k = 'training/' + k
                            wandb_logger.wandb.log({k: v})
                        wandb_logger.wandb.log({'training/lr': optimizer.param_groups[0]['lr']})
                        wandb_logger.wandb.log({'training/lat 1m': torch.sum(errlat <= 1) / errlat.size(0)})
                        wandb_logger.wandb.log({'training/lon 1m': torch.sum(errlong <= 1) / errlong.size(0)})
                        wandb_logger.wandb.log({'training/rot 1': torch.sum(errR <= 1) / errR.size(0)})
                        if 'topk_err' in pred.keys():
                            wandb_logger.wandg.log({'training/T_1 err': err_topk[0] / (it+1)})
                            wandb_logger.wandg.log({'training/T_topk err': err_topk[1] / (it+1)})
                            wandb_logger.wandg.log({'training/R_1 err': err_topk[2] / (it+1)})
                            wandb_logger.wandg.log({'training/R_topk err': err_topk[3] / (it+1)})


                    else:
                        for k, v in losses.items():
                            writer.add_scalar('training/'+k, v, tot_it)
                        writer.add_scalar(
                            'training/lr', optimizer.param_groups[0]['lr'], tot_it)

            del pred, data, loss, losses

            results = 0
            if (stop or it == (len(train_loader) - 1)):
            # if it == 0:
                ################
                ###VALIDATION###
                ################
                with fork_rng(seed=conf.train.seed):
                    results = do_evaluation(
                        model, val_loader, device, loss_fn, metrics_fn,
                        conf.train, pbar=(rank == 0), wandb_logger=wandb_logger)
                if rank == 0:
                    str_results = [f'{k} {v:.3E}' for k, v in results.items()]
                    logger.info(f'[Validation] {{{", ".join(str_results)}}}')

                    if args.wandb:
                        for k, v in results.items():
                            k = 'val/' + k
                            wandb_logger.wandb.log({k: v})
                    else:
                        for k, v in results.items():
                            writer.add_scalar('val/'+k, v, tot_it)
                torch.cuda.empty_cache()  # should be cleared at the first iter

                ################
                ######TEST######
                ################
                # test & save every epoch
                if args.save_every_epoch:
                    test_basic(dataset, model, wandb_logger)
                    torch.cuda.empty_cache()  # should be cleared at the first iter

            if stop:
                break

        if rank == 0:
            state = (model.module if args.distributed else model).state_dict()
            checkpoint = {
                'model': state,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'conf': OmegaConf.to_container(conf, resolve=True),
                'epoch': epoch,
                'losses': losses_,
                'eval': results,
            }

            # cp_name = f'checkpoint_{epoch}' + ('_interrupted' if stop else '')
            # cp_name = f'checkpoint_last' + ('_interrupted' if stop else '')
            if args.save_every_epoch:
                cp_name = f'checkpoint_{epoch}' + ('_interrupted' if stop else '')
            else:
                cp_name = f'checkpoint_last' + ('_interrupted' if stop else '')

            logger.info(f'Saving checkpoint {cp_name}')
            cp_path = str(output_dir / (cp_name + '.tar'))
            torch.save(checkpoint, cp_path)
            if results[conf.train.best_key] < best_eval:
                best_eval = results[conf.train.best_key]
                logger.info(
                    f'New best checkpoint: {conf.train.best_key}={best_eval}')
                shutil.copy(cp_path, str(output_dir / 'checkpoint_best.tar'))
            delete_old_checkpoints(
                output_dir, conf.train.keep_last_checkpoints)
            del checkpoint

        epoch += 1
        lr_scheduler.step()

        # original test mode
        # test(model, test_loader, wandb_logger=wandb_logger)
        if args.test_every_epoch:
            test(rank, conf, output_dir, args, wandb_logger=wandb_logger)

    if not args.test_every_epoch:
        test(rank, conf, output_dir, args, wandb_logger=wandb_logger)

    logger.info(f'Finished training on process {rank}.')
    if rank == 0 and wandb_logger == None:
        writer.close()


def main_worker(rank, conf, output_dir, args):

    if args.wandb:
        wandb_config = dict(project="cvl_fusion", entity='kaist-url-ai28', name=args.experiment)
        wandb_logger = WandbLogger(wandb_config, args)
        wandb_logger.before_run()
    else:
        wandb_logger = None

    if rank == 0:
        with capture_outputs(output_dir / 'log.txt'):
            if args.test:
                _ = test(rank, conf, output_dir, args, wandb_logger)
            else:
                training(rank, conf, output_dir, args, wandb_logger)
    else:
        if args.test:
            _ = test(rank, conf, output_dir, args, wandb_logger)
        else:
            training(rank, conf, output_dir, args, wandb_logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='kitti')
    parser.add_argument('--conf', type=str)
    parser.add_argument('--overfit', action='store_true', default=False)
    parser.add_argument('--restore', action='store_true', default=False)
    parser.add_argument('--finetune', type=str, default=False)
    parser.add_argument('--save_every_epoch', action='store_true', default=False, help='test & save every epoch')
    parser.add_argument('--test_every_epoch', action='store_true', default=False, help='test every epoch')
    parser.add_argument('--distributed', action='store_true',default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--dotlist', nargs='*', default=["data.name=kitti","data.max_num_points3D=4096","data.force_num_points3D=True",
                                                         "data.num_workers=4","data.train_batch_size=1","data.test_batch_size=1",
                                                         "data.rot_range=10", "data.trans_range=10", "data.aug=none", "data.aug_severity=3",
                                                         "train.lr=1e-5","model.name=two_view_refiner",
                                                         "model.optimizer.max_num_points3D=4096"])
    # parser.add_argument('--aug', type=str, default='none', choices=['none', 'aug1', 'augmix1', 'augmix1.g'])
    # parser.add_argument('--aug_severity', type=float, default=3)

    parser.add_argument('--wandb', action='store_true', default=False)

    args = parser.parse_intermixed_args()

    logger.info(f'Starting experiment {args.experiment}')
    output_dir = Path(TRAINING_PATH, args.experiment)
    output_dir.mkdir(exist_ok=True, parents=True)

    conf = OmegaConf.from_cli(args.dotlist)
    if args.conf:
        conf = OmegaConf.merge(OmegaConf.load(args.conf), conf)
    if not args.restore:
        #if conf.train.seed is None:
        conf.train.seed = torch.initial_seed() & (2**32 - 1) #default_train_conf.seed
        OmegaConf.save(conf, str(output_dir / 'config.yaml'))

    if args.distributed:
        args.n_gpus = torch.cuda.device_count()
        # os.environ["CUDA_VISIBLE_DEVICES"] = '3,4'
        os.environ["MASTER_ADDR"] = 'localhost'
        os.environ["MASTER_PORT"] = '1250'

        # debug by shan
        # os.environ["NCCL_DEBUG"] = 'INFO'
        # os.environ["NCCL_DEBUG_SUBSYS"] = 'ALL'
        # os.environ["NCCL_LL_THRESHOLD"] = '0'
        #os.environ["NCCL_BLOCKING_WAIT"] = '1'
        # os.environ["TORCH_DISTRIBUTED_DEBUG"] = 'INFO'

        args.lock_file = output_dir / "distributed_lock"
        if args.lock_file.exists():
            args.lock_file.unlink()

        torch.multiprocessing.spawn(
            main_worker, nprocs=args.n_gpus,
            args=(conf, output_dir, args))
    else:
        # os.environ["CUDA_VISIBL366E_DEVICES"] = '0'
        main_worker(0, conf, output_dir, args)
