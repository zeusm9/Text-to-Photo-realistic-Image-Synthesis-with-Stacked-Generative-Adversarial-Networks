import os
import torch
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter
from utils import mkdir_p

import config as cfg
from model import Stage1_G, Stage1_D
from utils import weights_init

class GANTrainer(object):
    def __init__(self,output_dir):
        if cfg.TRAIN_FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = SummaryWriter()

        self.max_epoch = cfg.TRAIN_MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN_SNAPSHOT_INTERVAL

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.batch_size = cfg.TRAIN_BATCH_SIZE * self.num_gpus
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

    def load_network_stageI(self):
        netG = Stage1_G()
        netG.apply(weights_init)
        print(netG)
        netD = Stage1_D()
        netD.apply(weights_init)
        print(netD)

        if cfg.CUDA:
            netG.cuda()
            netD.cuda()
        return netG,netD

    def train(self,data_loader,stage = 1):
        if stage == 1:
            netG,netD = self.load_network_stageI()