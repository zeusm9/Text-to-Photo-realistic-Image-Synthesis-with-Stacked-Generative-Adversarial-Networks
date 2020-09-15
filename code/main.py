import argparse
import random
import torch, pprint, datetime, dateutil.tz
import torchvision.transforms as transforms
import pickle
import numpy as np

import config as cfg
from datasets import TextDataset
from  train import GANTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train GAN network")
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
                 (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    if cfg.TRAIN_FLAG:
        image_transform = transforms.Compose([
            transforms.RandomCrop(cfg.IMSIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = TextDataset(cfg.DATA_DIR, 'train', imsize=cfg.IMSIZE, transform=image_transform,
                              embedding_filename=cfg.EMBEDDING_FILENAME)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN_BATCH_SIZE,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS)
        )
        algo = GANTrainer(output_dir)
        algo.train(data_loader,cfg.STAGE)
    else:
        image_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = TextDataset(cfg.DATA_DIR, 'test', imsize=cfg.IMSIZE, transform=image_transform,
                           embedding_filename=cfg.EMBEDDING_FILENAME)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN_BATCH_SIZE,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS)
        )
        N = len(dataloader)
        algo = GANTrainer(output_dir)
        algo.sample(dataloader, cfg.STAGE)

        dataset_generated = TextDataset(cfg.DATA_DIR, 'test', imsize=cfg.IMSIZE, transform=image_transform,
                              embedding_filename=cfg.EMBEDDING_FILENAME)
        dataloader_generated = torch.utils.data.DataLoader(
            dataset_generated, batch_size=cfg.TRAIN_BATCH_SIZE,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS)
        )
        print(algo.inception_score(dataloader_generated, batch_size=32, resize=True,splits=10))