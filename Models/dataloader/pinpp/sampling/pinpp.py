import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import math
import torch
import os

class PINPP(Dataset):

    def __init__(self, setname, args):
        IMAGE_PATH = os.path.join(args.data_dir, 'pinpp/images')
        SPLIT_PATH = os.path.join(args.data_dir, 'pinpp/split')
        ANNO_PATH = os.path.join(args.data_dir, 'pinpp/annotations')
        anno_txt_path = osp.join(ANNO_PATH, setname + '.txt')
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        data = []
        label = []
        lb = -1

        if 'num_patch' not in vars(args).keys():
            print ('do not assign num_patch parameter, set as default: 9')
            self.num_patch=9
        else:
            self.num_patch=args.num_patch

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data#data path of all data
        self.label = label #label of all data
        self.num_class = len(set(label))
        image_size = 84
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                 np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
        ])

        # process annotations
        self.bbox = []
        with open(anno_txt_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                # int_list = list(math.ceil(map(float, line.strip().split())))
                int_list = [math.ceil(float(x)) for x in line.strip().split()]
                int_list = [max(0, x) for x in int_list]
                self.bbox.append(int_list)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        patch_list=[]
        patch_list.append(self.transform(Image.open(path).convert('RGB').crop((self.bbox[i][0], self.bbox[i][1], self.bbox[i][0] + self.bbox[i][2], self.bbox[i][1] + self.bbox[i][3]))))

        # for _ in range(min(self.num_patch, len(self.bbox[i]) // 4)):
        for _ in range(self.num_patch - 1):
            patch_list.append(self.transform(Image.open(path).convert('RGB')))
            # patch_list.append(self.transform(Image.open(path).convert('RGB')[int(self.bbox[i][_*4+0]) : int(self.anno[i][_*4+0] + self.anno[i][_*4+2])][int(self.bbox[i][_*4+1]) : int(self.anno[i][_*4+1] + self.anno[i][_*4+3])]))
    
        while len(patch_list) < self.num_patch :
            patch_list.append(self.transform(Image.open(path).convert('RGB')))

        patch_list=torch.stack(patch_list,dim=0)
        return patch_list, label

if __name__ == '__main__':
    pass