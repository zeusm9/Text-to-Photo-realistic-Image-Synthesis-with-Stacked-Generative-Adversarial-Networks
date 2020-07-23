import torch.utils.data as data
import os.path
import PIL
import pickle
import numpy as np
import random

from PIL import Image

import config as cfg


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train', embedding_filename=cfg.EMBEDDING_FILENAME, imsize=64, transform=None,
                 target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.imsize = imsize
        self.data = []
        self.data_dir = data_dir
        split_dir = os.path.join(data_dir, split)

        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(split_dir, embedding_filename)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))

    def get_img(self, img_path):
        img = Image.open(img_path).convert('RGB')
        load_size = int(self.imsize * 76 / 64)
        img = img.resize((load_size, load_size), PIL.Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def load_embedding(self, data_dir, embedding_filename):
        with open(data_dir + embedding_filename, 'rb') as f:
            embeddings = pickle.load(f, encoding='bytes')
            embeddings = np.array(embeddings)
            print('embeddings: ', embeddings.shape)
        return embeddings

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id

    def __getitem__(self, index):
        key = self.filenames[index]
        data_dir = self.data_dir
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        img = self.get_img(img_name)

        embedding_ix = random.randint(0, embeddings.shape[0] - 1)
        embedding = embeddings[embedding_ix, :]
        if self.target_transform is not None:
            embedding = self.target_transform(embedding)
        return img, embedding

    def __len__(self):
        return len(self.filenames)
