'''
This version of DoReFa-Net uses PatchCamelyon dataset for training, validation, testing
in testing (inference) phase, convert images to grayscale
'''

import os
import time
import h5py
import argparse
import sklearn
from datetime import datetime
from PIL import Image

import torch
import torch.optim as optim
import torch.utils as utils
import torch.backends.cudnn as cudnn
from torchsummary import summary

cudnn.benchmark = True
import torchvision
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from nets.cifar_resnet import *

from utils.preprocessing import *

print('torch version: ', torch.__version__)

# Training settings
parser = argparse.ArgumentParser(description='DoReFa-Net pytorch implementation for PatchCamelyon')

parser.add_argument('--root_dir', type=str, default='out/')
#parser.add_argument('--data_dir', type=str, default='../data')
parser.add_argument('--log_name', type=str, default='resnet_w1a32')
parser.add_argument('--pretrain_dir', type=str, default='./ckpt/resnet20_baseline')

parser.add_argument('--exp_id', type=int, default=0) #logging
parser.add_argument('--Wbits', type=int, default=32)
parser.add_argument('--Abits', type=int, default=32)
parser.add_argument('--Dropout', action='store_true')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--color_mode', type=str, default='rgb') #rgb, grayscale, sp
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--eval_batch_size', type=int, default=1)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--model', type=str, default='resnet20')
parser.add_argument('--log_interval', type=int, default=5000)
#parser.add_argument('--use_gpu', type=str, default='') #empty means use no gpu
parser.add_argument('--use_gpu', type=str, default='0')
parser.add_argument('--num_workers', type=int, default=1)

parser.add_argument('--cluster', action='store_true', default=False)

cfg = parser.parse_args()
print('parser: ', parser)
print('arguments: ', cfg)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.use_gpu

out_path = os.path.join('out', str(cfg.exp_id))
while(os.path.exists(out_path)):
    #print(out_path, ' exists')
    cfg.exp_id += 1
    out_path = os.path.join('out', str(cfg.exp_id))
print('new out_path: ', out_path)
if(not os.path.exists(out_path)):
    print(out_path, ' not exists')
    os.mkdir(out_path)

cfg.log_dir = os.path.join(cfg.root_dir, str(cfg.exp_id), 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, str(cfg.exp_id), 'ckpt', cfg.log_name)
print('log dir: ', cfg.log_dir)

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

# modified from: https://github.com/alexmagsam/metastasis-detection/blob/master/data.py
class myDataset(torch.utils.data.Dataset):
    def __init__(self, path, mode): #later add augmentation [DONE]
        super().__init__()
        assert mode in ['train', 'valid', 'test']
        if(cfg.color_mode == 'grayscale'):
            base_name = '{}_grayscale_{}.npy'
            self.X = np.load(os.path.join(path, base_name.format(mode, 'x')))
            self.y = np.load(os.path.join(path, base_name.format(mode, 'y')))
            print('loaded data shapes: ', self.X.shape, self.y.shape)
        elif(cfg.color_mode == 'sparsity'):
            base_name = '{}_sparsity_{}.npy'
            self.X = np.load(os.path.join(path, base_name.format(mode, 'x')))
            self.y = np.load(os.path.join(path, base_name.format(mode, 'y')))
        elif(cfg.color_mode == 'grayscale_sparsity'):
            base_name = '{}_grayscale_sparsity_{}.npy'
            self.X = np.load(os.path.join(path, base_name.format(mode, 'x')))
            self.y = np.load(os.path.join(path, base_name.format(mode, 'y')))
        else:
            base_name = "camelyonpatch_level_2_split_{}_{}.h5"
            #num_white_images = 0
            print("# " * 10)
            print('Loading {} dataset...'.format(mode))
            # Open the files
            x_filename = os.path.join(path, base_name.format(mode, 'x'))
            print('trying to open file: ', x_filename)
            h5X = h5py.File(x_filename, 'r')
            h5y = h5py.File(os.path.join(path, base_name.format(mode, 'y')), 'r')
            print('x h5py: ', h5X)
            # Read into numpy array
            self.X = np.array(h5X.get('x'))
            self.y = np.array(h5y.get('y'))
        self.X = self.X.astype('uint8')
        print('X data shape: ', self.X.shape, ' type: ', self.X.dtype)
        if(self.X.shape[-1] != 3):
            raise ValueError('ERROR: data shape is not correct. It should be b,h,w,c NOT b,c,h,w')
        self.y = np.squeeze(self.y)
        print('Y data shape: ', self.y.shape, ' type: ', self.y.dtype)
        if(mode == 'train'):
            self.X, self.y = sklearn.utils.shuffle(self.X, self.y)
            print('done shuffling dataset')
        x_filename = os.path.join(path, base_name.format(mode, 'x'))
        y_filename = os.path.join(path, base_name.format(mode, 'y'))
        #np.save(file=x_filename, arr=self.X)
        #np.save(file=y_filename, arr=self.y)
        #print('np data files saved in', x_filename)
        print('Loaded {} dataset with {} samples'.format(mode, len(self.X)))
        print("# " * 10)
        #TODO: add color jitter later
        #print('performing random data augmentation: ho-or-ver flip, rotation(90)')
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                            #transforms.RandomHorizontalFlip(),
                                            #transforms.RandomVerticalFlip(),
                                            #transforms.RandomRotation(90),
                                            transforms.ToTensor()])
    
    def __getitem__(self, item):
        idx = item % self.__len__()
        #_slice = slice(idx*self.batch_size, (idx + 1) * self.batch_size)
        #image = self.X[idx]
        #print('images shape = ', images.shape, ' dtype: ', images.dtype)

        image = self.transform(self.X[idx])
        label = torch.tensor(self.y[idx].astype(np.int64))
        #print('images shape = ', images.shape, ' dtype: ', images.dtype)
        #images = torch.transpose(images, 0, 2).type(torch.FloatTensor)# .astype(np.float32)
        #print('images shape = ', images.shape, ' dtype: ', images.dtype)
        return {'images': image, 'labels': label}

    #def _transform(self, images):
    #    tensors = []
    #    for image in images:
    #        tensors.append(self.transform(image))
    #    return torch.stack(tensors)

    def __len__(self):
        return (len(self.X))



def main():
  data_path = '../data/camelyon/'
  valid_data = myDataset(data_path, mode='valid')
  valid_data_loader = utils.data.DataLoader(valid_data, batch_size=cfg.eval_batch_size, shuffle=False)
  print('valid dataset loader: ', valid_data_loader)
  train_data = myDataset(data_path, mode='train')
  test_data = myDataset(data_path, mode='test')
  print('train_data len: ', len(train_data))
  #print('train_data[0] images shape: ', train_data[0]['images'].shape)
  train_data_loader = utils.data.DataLoader(train_data, batch_size=cfg.train_batch_size, shuffle=True)
  test_data_loader = utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
  train_data = train_data_loader
  valid_data = valid_data_loader
  test_data = test_data_loader
  print('train data len: ', len(train_data))
  #print('train data[0] len: ', len(train_data[0]), ' type: ', type(train_data[0]), ' keys: ', train_data[0].keys())
  #sample_img = train_data[0]['images'][0]
  #print('sample img shape: ', sample_img.shape)
  #print('sample img max: ', torch.max(sample_img))
  #print('sample img mean: ', torch.mean(sample_img))
  #print('sample img min: ', torch.min(sample_img))
  #print('valid data shapes: ', valid_x.shape, valid_y.shape)
  #print('test data shapes: ', test_x.shape, test_y.shape)
  
  print('==> Building ResNet..')
  print('w-bits=', cfg.Wbits, ' a-bits=', cfg.Abits)
  if(cfg.model == 'resnet20'):
      model = resnet20(wbits=cfg.Wbits, abits=cfg.Abits, num_classes=2, dropout=cfg.Dropout).cuda()
  elif(cfg.model == 'resnet56'):
      model = resnet56(wbits=cfg.Wbits, abits=cfg.Abits, num_classes=2, dropout=cfg.Dropout).cuda()
  print(model)
  print('===== printing model summary ====')
  summary(model, (3, 96, 96))
  optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.wd)
  #optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd) 
  #Adam defaults: betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
  lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[cfg.max_epochs//5], gamma=0.1, verbose=True)
  criterion = torch.nn.CrossEntropyLoss().cuda()
  summary_writer = SummaryWriter(cfg.log_dir)
  print('lr scheduler: ', lr_schedu)
  print('optimizer: ', optimizer.state_dict())

  # Training
  def train(epoch):
    model.train()
    train_loss_list = []
    train_batch_time_list = []
    #print('training will loop for {} at this epoch'.format(len(train_data)))
    for batch_idx, sample in enumerate(train_data):
      #print('loop {}/{}'.format(batch_idx,len(train_data)))
      inputs = sample['images']
      targets = sample['labels']
      if(batch_idx == 0 and epoch % 10 == 0):
          print('iteration={}, inputs type={}, shape={}, inputs[0] max={}, min={}, \ntargets type={}, shape={}'.format(batch_idx, inputs.dtype, inputs.shape, torch.max(inputs[0]), torch.min(inputs[0]), targets.dtype, targets.shape))
          #img = transforms.ToPILImage()(inputs[0])
          #print('img type: ', type(img))
          #img_name = os.path.join(out_path, 'exp{}_train_img{}_class{}.png'.format(cfg.exp_id,
          #                        epoch, targets[0].item()))
          #img.save(img_name)
          #print('img saved in :', img_name)
      start_batch_train = time.time()
      optimizer.zero_grad()
         
      outputs = model(inputs.cuda()) #forward pass, outputs = yhat
      loss = criterion(outputs, targets.cuda())
      #print('loss = ', loss.item())
      if(np.isnan(loss.item())):
          raise ValueError('ERROR: loss value is NaN')
      train_loss_list.append(loss.item())
      loss.backward() #calculate gradients
      optimizer.step() #update parameters with gradient's value
      batch_train_time = time.time() - start_batch_train
      train_batch_time_list.append(batch_train_time)
      if (batch_idx % cfg.log_interval == 0 or batch_idx == cfg.max_epochs):
        step = len(train_data) * epoch + batch_idx
        #duration = time.time() - start_time
        #print('iteration={}, inputs type={}, inputs shape={}, targets type={}, targets shape={}'.format(batch_idx, inputs.dtype, inputs.shape, targets.dtype, targets.shape))
        #print('input max={}, min={}, targets={}'.format(torch.max(inputs), torch.min(inputs), targets.item()))
        print('epoch: %d step: %d/%d cls_loss= %.5f' %
              (epoch, batch_idx, len(train_data), loss.item()))
        #print('train time per batch list: length=', len(train_batch_time_list))
               #cfg.train_batch_size * cfg.log_interval / duration))
        #print('outputs shape={}, target shape={}'.format(outputs.shape, targets.shape))

        #start_time = time.time()
        summary_writer.add_scalar('cls_loss', loss.item(), step)
        summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], step)
    train_loss_per_epoch = np.mean(train_loss_list)
    train_time_per_epoch = np.sum(train_batch_time_list)
    #print('train epoch loss: ', train_epoch_loss)
    return train_loss_per_epoch, train_time_per_epoch

  
  def valid(epoch):
      print('====================== Validation ====================')
      #pass
      model.eval() #https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
      with torch.no_grad():
        valid_loss_list = []
        valid_batch_time_list = []
        for batch_idx, sample in enumerate(valid_data):
            inputs = sample['images']
            targets = sample['labels']
            #if(batch_idx == 0):
            #    print('iteration={}, inputs type={}, inputs shape={}, \ntargets type={}, targets shape={}'.format(batch_idx, inputs.dtype, inputs.shape, targets.dtype, targets.shape))
            #print('inputs shape={} type={}'.format(inputs.shape, type(inputs)))
            start_valid = time.time()
            outputs = model(inputs.cuda()) #forward pass
            loss = criterion(outputs, targets.cuda())
            valid_batch_time = time.time() - start_valid
            valid_batch_time_list.append(valid_batch_time)
            valid_loss_list.append(loss.item())
            if(batch_idx % 10000 == 0):
                print('iteration={}, output_shape={} loss={}, lenght of valid batch time list={}'.format(
                    batch_idx, outputs.shape, loss.item(), len(valid_batch_time_list)))
        valid_loss_per_epoch = np.mean(valid_loss_list)
        valid_time_per_epoch = np.sum(valid_batch_time_list)
        #print('epoch: %d valid_loss= %.5f' % (epoch, valid_epoch_loss))
      return valid_loss_per_epoch, valid_time_per_epoch

  def normal_test():
      print('======================= testing ==========================')
      # pass
      model.eval()
      #correct = 0
      #gs_correct = 0
      rgb_y_pred = []
      gs_y_pred = []
      sp_y_pred = []
      y_true = []
      rgb_time_list = []
      sp_time_list = []
      white_ratioes = []
      error_analysis_count = 20
      print('loop on test data for length = ', len(test_data))
      for batch_idx, sample in enumerate(test_data):
          inputs = sample['images']
          targets = sample['labels']
          if(batch_idx == 0):
              print('iter={}, inputs type={}, inputs shape={}, \ntargets type={}, targets shape={}'.format(batch_idx, inputs.dtype, inputs.shape, targets.dtype, targets.shape))
          inputs, targets = inputs.cuda(), targets.cuda()
          
          for k, an_input in enumerate(inputs):
              a_target = targets[k]
              #print('an input shape: ', an_input.shape)
              #print('a target shape: ', a_target.shape)
              img = transforms.ToPILImage()(an_input)
              #img = an_input.cpu().numpy()
              img = np.array(img)
              #print('img type: ', type(img), ' shape: ', img.shape, ' max: ', np.amax(img))
              #new_img = np.zeros(shape=(img.shape), dtype='uint8')
              #white_count = 0
              #loop over each pixel
              #for i in range(img.shape[0]):
              #    for j in range(img.shape[1]):
                      #print('img[{},{}], pixel={}'.format(i, j, img[i,j]))
              #        if(img[i,j, 0] >= 200 and img[i,j, 1] >= 200 and img[i,j, 2] >= 200):
                          #print('img[{},{}], white pixel={}'.format(i, j, img[i,j]))
                          #print('maybe white pixel')
              #            white_count += 1
              #            #leave new_image[i, j] = 0
              #        else:
              #            #insert original pixel in new image
              #            new_img[i, j] = img[i, j]
              #print('white count: ', white_count, ' total pixels: ', img.shape[0]*img.shape[1])
              #white_ratio = white_count/(img.shape[0]*img.shape[1])
              #if(white_ratio == 1.0):
              #    print('ERROR: white ratio = ', white_ratio)
              #    orig_img = Image.fromarray(img)
              #    trans_img = Image.fromarray(new_img)
              #    orig_img_name = os.path.join(out_path, 'exp{}_img_{}_class{}_orig_white.png'.format(cfg.exp_id,
              #                                           batch_idx, a_target.item()))
              #    trans_img_name = os.path.join(out_path, 'exp{}_img_{}_class{}_trans.png'.format(cfg.exp_id,
              #                                           batch_idx, a_target.item()))
              #    orig_img.save(orig_img_name)
              #    trans_img.save(trans_img_name)
                  #print('img saved: ', orig_img_name)

              #white_ratioes.append(white_ratio)

              #gs_input = transforms.Compose([transforms.ToPILImage()
              #                      ,transforms.Grayscale(num_output_channels=3)
              #                      ,transforms.ToTensor()
              #                      ])(an_input)

              an_input = torch.unsqueeze(an_input, dim=0)
              #gs_input = torch.unsqueeze(gs_input, dim=0)
              #sparsity_img = transforms.ToTensor()(new_img)
              #sp_input = torch.unsqueeze(sparsity_img, dim=0)
              #print('inputs shape: ', an_input.shape)
              #print('gs inputs shape: ', gs_input.shape)
              
              start_rgb = time.time()
              outputs = model(an_input)
              #print('rgb outputs shape: ', outputs.shape)
              #print('rgb output data: ', outputs.data)
              #pred = torcn.sigmoid(outputs)
              #print('pred shape: ', pred.shape, ' pred: ', pred)
              _, predicted = torch.max(outputs.data, 1)
              rgb_time_list.append(time.time() - start_rgb)

              #gs_outputs = model(gs_input.cuda())
              #_, gs_predicted = torch.max(gs_outputs.data, 1)
              
              #start_sp = time.time()
              #sparsity_output = model(sp_input.cuda())
              #_, sparsity_predicted = torch.max(sparsity_output, 1)
              #sp_time_list.append(time.time() - start_sp)
              if(predicted.item() != a_target.item() and error_analysis_count > 0):
                  print('getting error images, original label={}, predicted label={}'.format(a_target.item(),
                      predicted.item()))
                  #save two images: original and sparsity, each with its prediction
                  error_analysis_count -= 1
                  fig, (ax1, ax2) = plt.subplots(1, 2)
                  fig.suptitle('Original Label={}      Predicted={}'.format(a_target.item(), 
                      predicted.item()))
                  ax1.imshow(img)
                  #ax1.title('original')
              #    ax2.imshow(new_img)
                  #ax2.title('sparsity')
                  ax1.axis('off')
                  ax2.axis('off')
                  plt.show()
                  fig_path = '{}/exp_{}_orig_{}_error_{}.png'.format(out_path, 
                          cfg.color_mode,
                          cfg.exp_id, 
                          error_analysis_count)
                  plt.savefig(fig_path, dpi=200)
                  plt.close()
                  #print('fig saved: ', fig_path)
                  
              #if(batch_idx % 5000 == 0):
              #    print('y pred={}, y true={}'.format(predicted.item(), a_target.item()))
                  #print('white count: ', white_count, ' total pixels: ', img.shape[0]*img.shape[1])
                  #print('white ratio: ', white_ratio)
              
              rgb_y_pred.append(predicted.item())
              #gs_y_pred.append(gs_predicted.item())
              #sp_y_pred.append(sparsity_predicted.item())
              y_true.append(a_target.item())
              #correct += predicted.eq(a_target.data).cpu().sum().item()
              #gs_correct += gs_predicted.eq(a_target.data).cpu().sum().item()
              #print('RGB correct = {}, GS correct = {}'.format(correct, gs_correct))
          assert(len(rgb_y_pred) == len(y_true))
          if(batch_idx == 0):
              print('y lengths: ', len(rgb_y_pred), len(y_true))
      rgb_total_time = np.sum(rgb_time_list)
      print('Average time for testing per image: ', np.mean(rgb_time_list))
      #sp_total_time = np.sum(sp_time_list)
      #print('Average time for sp testing per image: ', np.mean(sp_time_list))
      #print('total time for sparsity testing:', sp_total_time)
      #print('Average of white ratioes: ', np.mean(white_ratioes))
      #print('Max of white ratioes: ', np.amax(white_ratioes))
      #print('Min of white ratioes: ', np.amin(white_ratioes))
      rgb_y_pred = np.array(rgb_y_pred)
      #gs_y_pred = np.array(gs_y_pred)
      #sp_y_pred = np.array(sp_y_pred)
      y_true = np.array(y_true)
      return rgb_y_pred, y_true, rgb_total_time

  '''
  def test():
    print(' ------------------------ modified test -------------------------')
    # pass
    model.eval()
    correct = 0
    y_pred = []
    y_true = []
    max_list = []
    min_list = []
    with torch.no_grad():
        print('looping on test data for {} iterations'.format(len(test_data)))
        for batch_idx, sample in enumerate(test_data):
            inputs = sample['images']
            targets = sample['labels']
            if(batch_idx == 0):
                print('iteration={}, inputs type={}, inputs shape={}, targets type={}, targets shape={}'.format(batch_idx, inputs.dtype, inputs.shape, targets.dtype, targets.shape))
            #white_count = 0 #count of white pixel in an image
            inputs, targets = inputs.cuda(), targets.cuda()
            pil_img = transforms.Compose([transforms.ToPILImage()
                     #, transforms.Grayscale(num_output_channels=3)
                     ])(inputs[0])
            #print('grayscale inputs type: ', type(grayscale_input))
            img = np.array(pil_img)
            #images = np.swapaxes(images, axis1=0, axis2=2)
            #images = np.expand_dims(images, axis=0)
            #images = np.expand_dims(images, axis=0).astype('float32')
            #print(' dtype: ', images.dtype)
            #print(' shape: ', images.shape)
            #images = torch.from_numpy(images).cuda()
            #image0 = image0.convert('RGB')
            #image0 = image0.resize(size=(image0.size[0]-6, image0.size[1]-6))
            #img = np.array(image0)
            new_img = np.zeros(shape=(img.shape), dtype='uint8')
            white_count = 0
            #loop over each pixel
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    print('img[{},{}], pixel={}'.format(i, j, img[i,j]))
                    if(img[i,j, 0] >= 200 and img[i,j, 1] >= 200 and img[i,j, 2] >= 200):
                        print('img[{},{}], pixel={}'.format(i, j, img[i,j]))
                        print('maybe white pixel')
                        white_count += 1
                        #leave new_image[i, j] = 0
                    else:
                        new_img[i, j] = img[i, j]
            outputs = model(images)
            print('output shape: ', outputs.shape)
            _, predicted = torch.max(outputs.data, 1)
            print('predicted (y_pred[i]): ', predicted.item())
            y_pred.append(predicted.item())
            y_true.append(targets.item())
            print('y_pred: ', y_pred)
            correct += predicted.eq(targets.data).cpu().sum().item()
            if(batch_idx % 4000 == 0):
                print('============ iteration # ', batch_idx, ' ===================')
                print('test inputs shape:', inputs.shape, ' targets shape: ', targets.shape)
                #print('image0 size: ', image0.size, ' np image shape: ', img.shape, 
                #        ' new image shape: ', new_img.shape)
                print('output shape: ', outputs.shape, ' type: ', type(outputs))
                print('predicted (y_pred[i]): ', predicted.item())
                print('targets (y_true[i]): ', targets.item())
                #print('white pixels count: {}. total pixel count = {}x{}={}'.format(white_count, img.shape[0],img.shape[1], img.shape[0]*img.shape[1]))
                #orig_img = transforms.ToPILImage()(inputs[0]) #convert to PIL
                orig_img = pil_img
                print('orig img type: ', type(orig_img))
                #orig_img = orig_img.numpy() #convert to numpy
                trans_img = Image.fromarray(new_img)
                print('img max: ', np.amax(orig_img), ' new img max: ', np.amax(trans_img))
                print('img min: ', np.min(orig_img), ' new img min: ', np.min(trans_img))
                #print('type of orig img: ', type(orig_img))
                #print('type trans img: ', type(trans_img))
                orig_img_name = os.path.join(out_path, 'exp{}_img_{}_class{}_orig.png'.format(cfg.exp_id, 
                    batch_idx, targets.item()))
                trans_img_name = os.path.join(out_path, 'exp{}_img_{}_class{}_trans.png'.format(cfg.exp_id, 
                    batch_idx, targets.item()))
                orig_img.save(orig_img_name)
                trans_img.save(trans_img_name)
                print('img saved: ', orig_img_name)
                print('img saved: ', trans_img_name)

        acc = 100. * correct / len(test_data)
        print('%s------------------------------------------------------ \n'
          'Test Precision@1: %.2f%% \n' % (datetime.now(), acc))
        summary_writer.add_scalar('Precision@1', acc)
    #print('y pred list type: ', type(y_pred))
    #print('y true list type: ', type(y_true))
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    #print('y pred shape: ', y_pred.shape)
    #print('y true shape: ', y_true.shape)
    return acc, y_pred, y_true
    '''

  train_losses = []
  valid_losses = []
  train_times = []
  valid_times = []
  #differences = []
  #if(cfg.max_epochs == 100):
  #    patience = cfg.max_epochs//10
  #else:
  #    patience = cfg.max_epochs
  #diff_times = 0
  for epoch in range(cfg.max_epochs):
    print('=========================================================')
    train_loss, train_epoch_time = train(epoch)
    valid_loss, valid_epoch_time = valid(epoch)
    #print('train loss = {}, train epoch time = {}\nvalid loss = {}, valid epoch time = {}'.format(
    #    train_loss, train_epoch_time, valid_loss, valid_epoch_time))
    #difference = abs(train_loss - valid_loss)
    #print('difference b/w train and valid loss = ', difference)
    #differences.append(difference)
    # if difference is increasing, stop training...
    #if(epoch > 1 and differences[epoch] > differences[epoch-1]):
    #    print('difference is increasing..., maybe overfitting')
    #    diff_times += 1
    #    print('increasing times: ', diff_times)
    #else:
        #reset patience
    #    diff_times = 0

    #if(diff_times == patience):
    #    print('STOP TRAINING: early stopping at epoch: {}/{} '.format(epoch, cfg.max_epochs))
    #    break
    lr_schedu.step() #maybe stepLR
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_times.append(train_epoch_time)
    valid_times.append(valid_epoch_time)
  #print('train valid loss curve figure saved')
  torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, 'checkpoint.t7'))
  #print('train losses: ', train_losses)
  #print('valid losses: ', valid_losses)
  #print('average train loss per epoch: ', np.mean(train_losses))
  #print('average valid loss per epoch: ', np.mean(valid_losses))
  print('average train time per epoch: ', np.mean(train_times))
  print('average valid time per epoch: ', np.mean(valid_times))
  plt.figure()
  plt.plot(np.log(train_losses), 'g--', label='training loss')
  plt.plot(np.log(valid_losses), '-', label='validation loss')
  plt.title('Training and Validation Loss, exp{}'.format(cfg.exp_id))
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(loc='upper right')
  fig_path = '{}/exp_{}_train_valid_loss.png'.format(out_path, cfg.exp_id)
  plt.savefig(fig_path, dpi=300)
  plt.close()
  print('train valid loss curve figure saved in path: ', fig_path)

  rgb_y_pred, y_true, rgb_test_time = normal_test()
  if(rgb_y_pred.shape != y_true.shape):
      print('rgb_y.shape={}, y_true.shape={}'.format(gs_y_pred.shape, y_true.shape))
      raise ValueError('ERROR: y shapes differ')
  print(cfg.color_mode, ' total testing time: ', rgb_test_time, ' average= ', rgb_test_time/len(test_data))
  #print('Sparsity total testing time: ', sp_test_time, ' average= ', sp_test_time/len(test_data))
  #test_accuracy, y_pred, y_true = test()
  from sklearn import metrics
  # 1: RGB testing results
  rgb_test_accuracy = metrics.accuracy_score(y_pred=rgb_y_pred, y_true=y_true)
  print('Test accuracy: ', rgb_test_accuracy)
  rgb_f1 = metrics.f1_score(y_pred=rgb_y_pred, y_true=y_true)
  print('F1 score :', rgb_f1)
  cm = metrics.confusion_matrix(y_pred=rgb_y_pred, y_true=y_true)
  #print('rgb confusion matrix')
  #print(cm)
  #print('rgb classification report')
  #print(metrics.classification_report(y_true=y_true, y_pred=rgb_y_pred, digits=3))
  # 2: Grayscale testing results
  #gs_f1 = metrics.f1_score(y_pred=gs_y_pred, y_true=y_true)
  #print('GS F1 score :', gs_f1)
  #cm2 = metrics.confusion_matrix(y_pred=gs_y_pred, y_true=y_true)
  #print('gs confusion matrix')
  #print(cm2)
  #print('gs classification report')
  #print(metrics.classification_report(y_true=y_true, y_pred=gs_y_pred, digits=3))
  # 3: Sparsity testing results
  #sp_test_accuracy = metrics.accuracy_score(y_pred=sp_y_pred, y_true=y_true)
  #sp_f1 = metrics.f1_score(y_pred=sp_y_pred, y_true=y_true)
  #print('SP F1 score :', sp_f1)
  #cm3 = metrics.confusion_matrix(y_pred=sp_y_pred, y_true=y_true)
  #print('sp confusion matrix')
  #print(cm3)
  #print('sp classification report')
  #print(metrics.classification_report(y_true=y_true, y_pred=sp_y_pred, digits=3))

  import seaborn as sns
  sns.set(font_scale=1.0) #label size
  ax = sns.heatmap(cm, annot=True, fmt="d",cmap='Greys')
  title = cfg.color_mode + ' Testing Accuracy=' + str(np.around(rgb_test_accuracy, decimals=3))
  plt.title(title)
  plt.xlabel('Predicted Classes')
  plt.ylabel('True Classes')
  plt.show()
  img_name = '{}/exp_{}_cm.png'.format(out_path, cfg.exp_id)
  plt.savefig(img_name, dpi=200)
  print('image saved in ', img_name)
  plt.close()
  summary_writer.close()

  #sns.set(font_scale=1.0) #label size
  #ax = sns.heatmap(cm3, annot=True, fmt="d",cmap='Greys')
  #title = 'Sparsity Testing Accuracy=' + str(np.around(sp_test_accuracy, decimals=3))
  #plt.title(title)
  #plt.xlabel('Predicted Classes')
  #plt.ylabel('True Classes')
  #plt.show()
  #img_name = '{}/exp_{}_sp_cm.png'.format(out_path, cfg.exp_id)
  #plt.savefig(img_name, dpi=300)
  #print('image saved in ', img_name)
  #plt.close()
  


if __name__ == '__main__':
  main()
