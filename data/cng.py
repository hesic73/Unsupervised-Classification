from torch.utils.data import Dataset
from utils.mypath import MyPath

import os
import json
import numpy as np
from PIL import Image

class CNG(Dataset):

  def __init__(self, root=MyPath.db_root_dir('cng'), split='train', transform=None):
    
    super(CNG, self).__init__()
    self.root = root
    self.transform = transform
    self.split=split
    self.data =[]
    self.label = []
    self.img_names = []
    self.q = []
    if self.split == 'train':
      self.meta = os.path.join(self.root, 'train.json')
      self.data_dir = os.path.join(self.root, 'all')
    else:
      self.meta = os.path.join(self.root, 'val.json')
      self.data_dir = os.path.join(self.root, 'all')
    
    self.mean = []
    self.std = []
    
    #self.mean, self.std = self.get_mean_and_var(self.data_dir, 64*64)

    with open(self.meta) as f:
      content = json.load(f)

      for item in content:
        fp = open(os.path.join(self.data_dir, item['img_name']), 'rb')
        im = Image.open(fp)
        im = np.asarray(im)
        #print(im.shape)
        self.data.append(im)
        self.label.append(item['class_id'])
        self.img_names.append(item['img_name'])
        #self.q.append(item["q"])
        fp.close()
    self.data = np.asarray(self.data) 
    #print(self.data.shape)
    #print(self.data.shape)
    #self.data = self.data.transpose((0, 2, 3, 1))
    self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
  def __getitem__(self, index):
    '''
    Args:
      index(int): Indx
    Returns:
      dict: {'image': image, 'target': index of target class, 'meta': dict}
    '''
        
    img, target = self.data[index], self.label[index]
    img_size = (img.shape[0], img.shape[1])
    #print(img[:,:,0] - img[:,:,1])
    img = Image.fromarray(img)
    
    img.save('./1.png')
    if self.transform is not None:
      img = self.transform(img)
    out = {'image': img, 'target': target, 'meta': {'im_size': img_size, 'index': index, 'class_name': self.classes[target], 'img_name': self.img_names[index]}}

    return out

  def get_image(self, index):
    img = self.data[index]
    return img

  def get_index_by_name(self, name):
    return self.img_names.index(name)
  
  def __len__(self):
    return len(self.data)
  
  
  @staticmethod  
  def get_mean_and_var(filepath, pixels):
    dir = os.listdir(filepath)
    #print(dir)
    r, g, b = 0, 0, 0
    for idx in range(len(dir)):
      filename = dir[idx]
      img = np.asarray(Image.open(os.path.join(filepath, filename))) / 255.0
      if len(img.shape) == 2:
        img = np.array([img, img, img]).reshape(160, 160, 3)
      r = r + np.sum(img[:, :, 0])
      g = g + np.sum(img[:, :, 1])
      b = b + np.sum(img[:, :, 2])
    
    pixels = len(dir) * pixels 
    r_mean = r / pixels
    g_mean = g / pixels
    b_mean = b / pixels

    r, g, b = 0, 0, 0
    for i in range(len(dir)):
      filename = dir[i]
      img = np.asarray(Image.open(os.path.join(filepath, filename))) / 255.0
      if len(img.shape) == 2:
        img = np.array([img, img, img]).reshape(160, 160, 3)
      r = r + np.sum((img[:, :, 0] - r_mean) ** 2)
      g = g + np.sum((img[:, :, 1] - g_mean) ** 2)
      b = b + np.sum((img[:, :, 2] - b_mean) ** 2)
    
    r_var = np.sqrt(r / pixels)
    g_var = np.sqrt(g / pixels)
    b_var = np.sqrt(b / pixels)
    r_mean = np.float32(r_mean)
    g_mean = np.float32(g_mean)
    b_mean = np.float32(b_mean)
    r_var = np.float32(r_var)
    g_var = np.float32(g_var)
    b_var = np.float32(b_var)
    print("r_mean is %f, g_mean is %f, b_mean is %f" % (r_mean, g_mean, b_mean))
    print("r_var is %f, g_var is %f, b_var is %f" % (r_var, g_var, b_var))
    return [r_mean, g_mean, g_mean], [r_var, g_var, b_var]

# test code

if __name__ == '__main__':
  dataset = CNG()
  x = dataset.data[0]
  im = Image.fromarray(x)
  im.save('./1.png')
