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

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--wd', type=float, default=1e-4)

parser.add_argument('--train_batch_size', type=int, default=16)
parser.add_argument('--eval_batch_size', type=int, default=100)
parser.add_argument('--max_epochs', type=int, default=100)

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
    def __init__(self, path, mode): #later add augmentation
        super().__init__()
        assert mode in ['train', 'valid', 'test']
        base_name = "camelyonpatch_level_2_split_{}_{}.h5"
        print("# " * 10)
        print('Loading {} dataset...'.format(mode))
        # Open the files
        h5X = h5py.File(os.path.join(path, base_name.format(mode, 'x')), 'r')
        h5y = h5py.File(os.path.join(path, base_name.format(mode, 'y')), 'r')
        print('x h5py: ', h5X)
        # Read into numpy array
        self.X = np.array(h5X.get('x'))
        self.y = np.array(h5y.get('y'))
        print('X data shape: ', self.X.shape, ' type: ', self.X.dtype)
        #print('Y data shape: ', self.y.shape, ' type: ', self.y.dtype)
        self.y = np.squeeze(self.y)
        print('Y data shape: ', self.y.shape, ' type: ', self.y.dtype)
        self.X, self.y = sklearn.utils.shuffle(self.X, self.y)
        print('done shuffling dataset')
        x_filename = os.path.join(path, base_name.format(mode, 'x'))
        y_filename = os.path.join(path, base_name.format(mode, 'y'))
        #np.save(file=x_filename, arr=self.X)
        #np.save(file=y_filename, arr=self.y)
        #print('np data files saved in', x_filename)
        print('Loaded {} dataset with {} samples'.format(mode, len(self.X)))
        print("# " * 10)
        self.transform = transforms.Compose([transforms.ToPILImage(), 
                                            transforms.ToTensor()])
    
    def __getitem__(self, item):
        idx = item % self.__len__()
        #_slice = slice(idx*self.batch_size, (idx + 1) * self.batch_size)
        images = self.X[idx]
        #print('images shape = ', images.shape, ' dtype: ', images.dtype)
        images = self.transform(self.X[idx])
        labels = torch.tensor(self.y[idx].astype(np.int64))
        #print('images shape = ', images.shape, ' dtype: ', images.dtype)
        #images = torch.transpose(images, 0, 2).type(torch.FloatTensor)# .astype(np.float32)
        #print('images shape = ', images.shape, ' dtype: ', images.dtype)
        return {'images': images, 'labels': labels}

    #def _transform(self, images):
    #    tensors = []
    #    for image in images:
    #        tensors.append(self.transform(image))
    #    return torch.stack(tensors)

    def __len__(self):
        return (len(self.X))



def main():
  data_path = '../../camelyon/data/'
  valid_data = myDataset(data_path, mode='valid')
  valid_data_loader = utils.data.DataLoader(valid_data, batch_size=cfg.eval_batch_size, shuffle=False)
  print('valid dataset loader: ', valid_data_loader)
  train_data = myDataset(data_path, mode='train')
  test_data = myDataset(data_path, mode='test')
  print('train_data len: ', len(train_data))
  print('train_data[0] images shape: ', train_data[0]['images'].shape)
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
  model = resnet20(wbits=cfg.Wbits, abits=cfg.Abits, num_classes=2).cuda()
  print(model)
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
    start_time = time.time()
    #print('training will loop for {} at this epoch'.format(len(train_data)))
    for batch_idx, sample in enumerate(train_data):
      #print('loop {}/{}'.format(batch_idx,len(train_data)))
      inputs = sample['images']
      targets = sample['labels']
      if(batch_idx % 10 == 0):
          print('iteration={}, inputs type={}, shape={}, inputs[0] max={}, min={}, targets type={}, shape={}'.format(batch_idx, inputs.dtype, inputs.shape, torch.max(inputs[0]), torch.min(inputs[0]), targets.dtype, targets.shape))
          img = transforms.ToPILImage()(inputs[0])
          print('img type: ', type(img))
          img_name = os.path.join(out_path, 'exp{}_train_img{}_class{}.png'.format(cfg.exp_id,
                                  epoch, targets[0].item()))
          img.save(img_name)
          print('img saved in :', img_name)
      optimizer.zero_grad()
         
      outputs = model(inputs.cuda()) #forward pass, outputs = yhat
      loss = criterion(outputs, targets.cuda())
      if(np.isnan(loss.item())):
          raise ValueError('ERROR: loss value is NaN')
      train_loss_list.append(loss.item())
      loss.backward() #calculate gradients
      optimizer.step() #update parameters with gradient's value
      
      if (batch_idx % cfg.log_interval == 0):
        step = len(train_data) * epoch + batch_idx
        duration = time.time() - start_time
        #print('iteration={}, inputs type={}, inputs shape={}, targets type={}, targets shape={}'.format(batch_idx, inputs.dtype, inputs.shape, targets.dtype, targets.shape))
        #print('input max={}, min={}, targets={}'.format(torch.max(inputs), torch.min(inputs), targets.item()))
        print('epoch: %d step: %d/%d cls_loss= %.5f (%d samples/sec)' %
              (epoch, batch_idx, len(train_data), loss.item(),
               cfg.train_batch_size * cfg.log_interval / duration))

        start_time = time.time()
        summary_writer.add_scalar('cls_loss', loss.item(), step)
        summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], step)
    train_epoch_loss = np.mean(train_loss_list)
    #print('train loss list: ', train_loss_list)
    print('train epoch loss: ', train_epoch_loss)
    return train_epoch_loss

  
  def valid(epoch):
      print('====================== Validation ====================')
      #pass
      model.eval() #https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
      with torch.no_grad():
        valid_loss_list = []
        for batch_idx, sample in enumerate(valid_data):
            inputs = sample['images']
            targets = sample['labels']
            if(batch_idx == 0):
                print('iteration={}, inputs type={}, inputs shape={}, targets type={}, targets shape={}'.format(batch_idx, inputs.dtype, inputs.shape, targets.dtype, targets.shape))
            #print('inputs shape={} type={}'.format(inputs.shape, type(inputs)))
            outputs = model(inputs.cuda()) #forward pass
            loss = criterion(outputs, targets.cuda())
            valid_loss_list.append(loss.item())
            if(batch_idx % 5000 == 0):
                print('iteration={}, output_shape={} loss={}'.format(batch_idx, outputs.shape, loss.item()))
        valid_epoch_loss = np.mean(valid_loss_list)
        print('epoch: %d valid_loss= %.5f' % (epoch, valid_epoch_loss))
      return valid_epoch_loss

  def normal_test():
      print('======================= normal test ==========================')
      # pass
      model.eval()
      correct = 0
      for batch_idx, sample in enumerate(test_data):
          inputs = sample['images']
          targets = sample['labels']
          if(batch_idx == 0):
              if(batch_idx == 0):
                    print('iteration={}, inputs type={}, inputs shape={}, targets type={}, targets shape={}'.format(batch_idx, inputs.dtype, inputs.shape, targets.dtype, targets.shape))
          inputs, targets = inputs.cuda(), targets.cuda()
          outputs = model(inputs)
          _, predicted = torch.max(outputs.data, 1)
          correct += predicted.eq(targets.data).cpu().sum().item()
      acc = 100. * correct / len(test_data)
      print('%s------------------------------------------------------ '
                'Precision@1: %.2f%% \n' % (datetime.now(), acc))
      summary_writer.add_scalar('Precision@1', acc, global_step=epoch)

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
            gs_pil_img = transforms.Compose([transforms.ToPILImage(), 
                     transforms.Grayscale(num_output_channels=3)])(inputs[0])
            #print('grayscale inputs type: ', type(grayscale_input))
            images = np.array(gs_pil_img)
            images = np.swapaxes(images, axis1=0, axis2=2)
            #images = np.expand_dims(images, axis=0)
            images = np.expand_dims(images, axis=0).astype('float32')
            #print(' dtype: ', images.dtype)
            #print(' shape: ', images.shape)
            images = torch.from_numpy(images).cuda()
            #image0 = image0.convert('RGB')
            #image0 = image0.resize(size=(image0.size[0]-6, image0.size[1]-6))
            #img = np.array(image0)
            #new_img = np.zeros(shape=(img.shape), dtype='uint8')
                                             
            #loop over each pixel
            #for i in range(img.shape[0]):
            #    for j in range(img.shape[1]):
                    #print('img[{},{}], pixel={}'.format(i, j, img[i,j]))
            #        if(img[i,j, 0] >= 200 and img[i,j, 1] >= 200 and img[i,j, 2] >= 200):
                        #print('img[{},{}], pixel={}'.format(i, j, img[i,j]))
                        #print('maybe white pixel')
            #            white_count += 1
            #        else:
            #            new_img[i, j] = img[i, j]
            outputs = model(images)
            #print('output shape: ', outputs.shape)
            _, predicted = torch.max(outputs.data, 1)
            #print('predicted (y_pred[i]): ', predicted.item())
            y_pred.append(predicted.item())
            y_true.append(targets.item())
            #print('y_pred: ', y_pred)
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
                orig_img = transforms.ToPILImage()(inputs[0]) #convert to PIL
                print('orig img type: ', type(orig_img))
                #orig_img = orig_img.numpy() #convert to numpy
                trans_img = gs_pil_img
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

  start_training = time.time()
  train_losses = []
  valid_losses = []
  differences = []
  if(cfg.max_epochs == 100):
      patience = cfg.max_epochs//10
  else:
      patience = cfg.max_epochs
  diff_times = 0
  for epoch in range(cfg.max_epochs):
    print('\nEpoch: %d =============' % epoch)
    train_loss = train(epoch)
    valid_loss = valid(epoch)
    difference = abs(train_loss - valid_loss)
    print('difference b/w train and valid loss = ', difference)
    differences.append(difference)
    # if difference is increasing, stop training...
    if(epoch > 1 and differences[epoch] > differences[epoch-1]):
        print('difference is increasing..., maybe overfitting')
        diff_times += 1
        print('increasing times: ', diff_times)
    else:
        #reset patience
        diff_times = 0

    if(diff_times == patience):
        print('STOP TRAINING: early stopping at epoch: {}/{} '.format(epoch, cfg.max_epochs))
        break
    lr_schedu.step() #maybe stepLR
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

  print('train losses: ', train_losses)
  print('valid losses: ', valid_losses)
  plt.figure()
  plt.plot(np.log(train_losses), 'g--', label='training loss')
  plt.plot(np.log(valid_losses), '-', label='validation loss')
  plt.title('Training and Validation Loss, exp{}'.format(cfg.exp_id))
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(loc='upper right')
  plt.savefig('{}/exp_{}_train_valid_loss.png'.format(out_path, cfg.exp_id), dpi=300)
  plt.close()
  end_training = time.time()
  print('train valid loss curve figure saved')
  print('training time: ', end_training - start_training, ' seconds')
  torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, 'checkpoint.t7'))

  normal_test()
  test_accuracy, y_pred, y_true = test()
  from sklearn.metrics import confusion_matrix, classification_report
  cm = confusion_matrix(y_pred=y_pred, y_true=y_true)
  print('confusion matrix')
  print(cm)
  print('classification report')
  print(classification_report(y_true, y_pred, digits=3))
  import seaborn as sns
  sns.set(font_scale=1.0) #label size
  ax = sns.heatmap(cm, annot=True, fmt="d",cmap='Greys')
  title = 'Testing Accuracy=' + str(np.around(test_accuracy, decimals=2))
  plt.title(title)
  plt.xlabel('Predicted Classes')
  plt.ylabel('True Classes')
  plt.show()
  img_name = '{}/exp_{}_cm.png'.format(out_path, cfg.exp_id)
  plt.savefig(img_name, dpi=600)
  print('image saved in ', img_name)
  plt.close()
  summary_writer.close()


if __name__ == '__main__':
  main()
