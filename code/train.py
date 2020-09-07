import os
import torch
import time
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter as summary
from utils import mkdir_p
from torch.autograd import Variable

import config as cfg
from model import Stage1_G, Stage1_D
from utils import weights_init, discriminator_loss, generator_loss, KL_loss, save_img_results, save_model


class GANTrainer(object):
    def __init__(self, output_dir):
        if cfg.TRAIN_FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = summary()

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

        if cfg.NET_G != '':
            state_dict = torch.load(cfg.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_G)
        if cfg.NET_D != '':
            state_dict = torch.load(cfg.NET_D, map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_D)

        if cfg.CUDA:
            netG.cuda()
            netD.cuda()
        return netG, netD

    def load_network_stageII(self):
        from model import Stage1_G, Stage2_G, Stage2_D
        Stage1_G = Stage1_G()
        netG = Stage2_G(Stage1_G)
        netG.apply(weights_init)
        print(netG)
        if cfg.NET_G != '':
            state_dict = \
                torch.load(cfg.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_G)
        elif cfg.STAGE1_G != '':
            state_dict = \
                torch.load(cfg.STAGE1_G,
                           map_location=lambda storage, loc: storage)
            netG.Stage1_G.load_state_dict(state_dict)
            print('Load from: ', cfg.STAGE1_G)
        else:
            print("Please give the Stage1_G path")
            return

        netD = Stage2_D()
        netD.apply(weights_init)
        if cfg.NET_D != '':
            state_dict = \
                torch.load(cfg.NET_D,
                           map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_D)
        print(netD)

        if cfg.CUDA:
            netG.cuda()
            netD.cuda()
        return netG, netD

    def train(self, data_loader, stage=1):
        if stage == 1:
            netG, netD = self.load_network_stageI()
        else:
            netG, netD = self.load_network_stageII()

        nz = cfg.Z_DIM
        batch_size = self.batch_size
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1), requires_grad=True)
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))

        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
            real_labels, fake_labels = real_labels.cuda(), fake_labels.cuda()

        generator_lr = cfg.TRAIN_GENERATOR_LR
        discriminator_lr = cfg.TRAIN_DISCRIMINATOR_LR
        lr_decay_step = cfg.TRAIN_LR_DECAY_EPOCH
        optimizerD = optim.Adam(netD.parameters(), lr=cfg.TRAIN_DISCRIMINATOR_LR, betas=(0.5, 0.999))

        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        optimizerG = optim.Adam(netG_para, lr=cfg.TRAIN_GENERATOR_LR, betas=(0.5, 0.999))

        count = 0
        for epoch in range(self.max_epoch):
            start_t = time.time()
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr
                discriminator_lr *= 0.5
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = discriminator_lr
            for i, data in enumerate(data_loader, 0):
                # Prepare training data
                real_img_cpu, txt_embedding = data
                real_imgs = Variable(real_img_cpu)
                txt_embedding = Variable(txt_embedding)
                if cfg.CUDA:
                    real_imgs = real_imgs.cuda()
                    txt_embedding = txt_embedding.cuda()
                # Generate fake images
                noise.data.normal_(0, 1)
                inputs = (txt_embedding, noise)
                _, fake_imgs, mu, logvar = nn.parallel.data_parallel(netG, inputs, self.gpus)

                # Update D network

                netD.zero_grad()
                errD, errD_real, errD_wrong, errD_fake = discriminator_loss(netD, real_imgs, fake_imgs,
                                                                            real_labels, fake_labels, mu,
                                                                            self.gpus)
                errD.backward()
                optimizerD.step()
                ############################
                # (2) Update G network
                ###########################
                netG.zero_grad()
                errG = generator_loss(netD, fake_imgs,
                                      real_labels, mu, self.gpus)
                kl_loss = KL_loss(mu, logvar)
                errG_total = errG + kl_loss * cfg.TRAIN_COEFF_KL
                errG_total.backward()
                optimizerG.step()

                count = count + 1

                if i % 100 == 0:

                    # save the image result for each epoch
                    inputs = (txt_embedding, fixed_noise)
                    lr_fake, fake, _, _ = \
                        nn.parallel.data_parallel(netG, inputs, self.gpus)
                    save_img_results(real_img_cpu, fake, epoch, self.image_dir)
                    if lr_fake is not None:
                        save_img_results(None, lr_fake, epoch, self.image_dir)
                end_t = time.time()
                print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_KL: %.4f
                                     Loss_real: %.4f Loss_wrong:%.4f Loss_fake %.4f
                                     Total Time: %.2fsec
                                  '''
                      % (epoch, self.max_epoch, i, len(data_loader),
                         errD.data, errG.data, kl_loss.data,
                         errD_real, errD_wrong, errD_fake, (end_t - start_t)))
                if epoch % self.snapshot_interval == 0:
                    save_model(netG, netD, epoch, self.model_dir)
                #
            save_model(netG, netD, self.max_epoch, self.model_dir)
            #
            self.summary_writer.close()
