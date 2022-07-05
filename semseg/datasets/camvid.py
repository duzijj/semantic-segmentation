from PIL import Image
import torch.utils.data as data
import os
from torchvision import transforms as T
import torch
import numpy as np
import time



def rgb_to_mask2(rgb, color_to_class):
    #torch.Size([720, 960, 3])
    rgb=np.array(rgb)
    W,H,C=rgb.shape
    #mask=torch.zeros(W,H,dtype=torch.uint8)
    mask=np.full((W,H),255,dtype=np.uint8)
    for color,cls in color_to_class.items():
        color=np.array(color, dtype=np.uint8)
        _mask=np.all(rgb==color,axis=2)
        mask[_mask]=cls
    mask=Image.fromarray(mask)
    return mask

class CamVid(data.Dataset):
    def __init__(self,root,split,transform):
        # images, masks, json splits
        self.n_classes = 11
        self.images=[]
        self.masks=[]
        self.ignore_label = 255
        root= os.path.expanduser(root)
        splits=[]
        if "train" in split:
            splits.append("train")
        if "val" in split:
            splits.append("val")
        if "test" in split:
            splits.append("test")
        for split in splits:
            image_dir=os.path.join(root,split)
            mask_dir=os.path.join(root,split+"_labels")
            for base_name in sorted(os.listdir(image_dir)):
                image_name=os.path.join(image_dir, base_name)
                self.images.append(image_name)
                mask_name=base_name[:-4]+"_L"+base_name[-4:]
                mask_name=os.path.join(mask_dir,mask_name)
                self.masks.append(mask_name)
        self.transform=transform
        assert (len(self.images) == len(self.masks))
        class_names=["Building","Tree","Sky","Car","SignSymbol","Road",
                     "Pedestrian","Fence","Column_Pole","Sidewalk","Bicyclist"]
        self.color_to_class={(0, 128, 192): 10,
                             (128, 0, 0): 0,
                             (64, 0, 128): 3,
                             (192, 192, 128): 8,
                             (64, 64, 128): 7,
                             (64, 64, 0): 6,
                             (128, 64, 128): 5,
                             (0, 0, 192): 9,
                             (192, 128, 128): 4,
                             (128, 128, 128): 2,
                             (128, 128, 0): 1}
    def build_epoch(self):
        # Cityscapes dataset needs build_epoch for class uniform sampling
        # Camvid does nothing here
        return

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        target = rgb_to_mask2(target, self.color_to_class)
        if self.transform is not None:
            img, target = self.transform(img,target)
        return img, target

    def __len__(self):
        return len(self.images)



if __name__=='__main__':
    import numpy as np
    dataset=CamVid(root='D:\DDRNET\data\camvid' ,split="train",transform=None)
    # dataset=Camvid("D:\DeepLabV3Plus\datasets\data\camvid_big",split="test",transforms=T.ToTensor())
    print(dataset[1])
    (np.array(dataset[1][1])==11).sum()
    # print(t4-t3)
    # print(image.shape)
    # print(target.shape)
    
