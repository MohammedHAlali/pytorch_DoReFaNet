import os
import time
import argparse
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

# Training settings
parser = argparse.ArgumentParser(description='DoReFa-Net pytorch implementation')

parser.add_argument('--root_dir', type=str, default='out/')
parser.add_argument('--data_dir', type=str, default='../data')
parser.add_argument('--log_name', type=str, default='resnet_w1a32')
parser.add_argument('--model', type=str, default='cifar10')
parser.add_argument('--pretrain_dir', type=str, default='./ckpt/resnet20_baseline')

parser.add_argument('--cifar', type=int, default=10)
parser.add_argument('--exp_id', type=int, default=0) #logging
parser.add_argument('--Wbits', type=int, default=1)
parser.add_argument('--Abits', type=int, default=32)

parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--wd', type=float, default=1e-4)

parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--eval_batch_size', type=int, default=1)
parser.add_argument('--max_epochs', type=int, default=10)

parser.add_argument('--log_interval', type=int, default=200)
#parser.add_argument('--use_gpu', type=str, default='') #empty means use no gpu
parser.add_argument('--use_gpu', type=str, default='0')
parser.add_argument('--num_workers', type=int, default=1)

parser.add_argument('--cluster', action='store_true', default=False)

cfg = parser.parse_args()
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

def main():
  if cfg.model == 'mnist':
      print('training MNIST')
      num_classes = 10
      dataset = torchvision.datasets.MNIST
      color_mode = 'L' #greyscale
  elif cfg.cifar == 10:
    print('training CIFAR-10 !')
    num_classes = 10
    dataset = torchvision.datasets.CIFAR10
    color_mode = 'rgb'
  elif cfg.cifar == 100:
    print('training CIFAR-100 !')
    num_classes = 100
    dataset = torchvision.datasets.CIFAR100
    color_mode = 'rgb'
  else:
    assert False, 'dataset unknown !'

  print('==> Preparing data ..')
  train_dataset = dataset(root=cfg.data_dir, train=True, download=True,
                          transform=cifar_transform(is_training=True))
  image0 = train_dataset[0][0]
  print('image0 shape: ', image0.shape)
  image0 = transforms.ToPILImage()(image0)
  image0 = np.array(image0)
  print('image0: type: ', type(image0))
  print('image0 shape: ', image0.shape)
  print('image0 max: ', np.amax(image0))
  print('image0 min: ', np.min(image0))
  #print('image0[0].shape: ', image0[0].shape)
  #exit()
  print('train dataset: ', train_dataset)
  print('dataset length: ', len(train_dataset))
  train_dataset, valid_dataset = utils.data.random_split(train_dataset, [len(train_dataset)-5000, 5000])
  print('train dataset: ', train_dataset)
  print('valid dataset: ', valid_dataset, ' length: ', len(valid_dataset))
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True,
                                             num_workers=cfg.num_workers)
  valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.train_batch_size, shuffle=True,
                                                       num_workers=cfg.num_workers)
  print('valid data_loader: ', valid_loader)
  test_dataset = dataset(root=cfg.data_dir, train=False, download=True,
                         transform=cifar_transform(is_training=False))
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.eval_batch_size, shuffle=False,
                                            num_workers=cfg.num_workers)
  print('test dataset: ', test_dataset)
  print('==> Building ResNet..')
  print('w-bits=', cfg.Wbits, ' a-bits=', cfg.Abits)
  model = resnet20(wbits=cfg.Wbits, abits=cfg.Abits, num_classes=num_classes).cuda()
  print(model)
  optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.wd)
  lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[cfg.max_epochs//2], gamma=0.1, verbose=True)
  criterion = torch.nn.CrossEntropyLoss().cuda()
  summary_writer = SummaryWriter(cfg.log_dir)
  print('lr scheduler: ', lr_schedu)
  print('optimizer: ', optimizer.state_dict())

  # Training
  def train(epoch):
    model.train()
    train_loss_list = []
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
      print('iteration: ', batch_idx, '*'*50)
      print('inputs shape={} type={}'.format(inputs.shape, type(inputs)))
      print('targets shape={} type={}'.format(targets.shape, type(targets)))
      outputs = model(inputs.cuda()) #forward pass, outputs = yhat
      loss = criterion(outputs, targets.cuda())
      train_loss_list.append(loss.item())
      optimizer.zero_grad()
      loss.backward() #calculate gradients
      optimizer.step() #update parameters with gradient's value

      if batch_idx % cfg.log_interval == 0:
        step = len(train_loader) * epoch + batch_idx
        duration = time.time() - start_time

        print('%s epoch: %d step: %d cls_loss= %.5f (%d samples/sec)' %
              (datetime.now(), epoch, batch_idx, loss.item(),
               cfg.train_batch_size * cfg.log_interval / duration))

        start_time = time.time()
        summary_writer.add_scalar('cls_loss', loss.item(), step)
        summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], step)
    train_epoch_loss = np.mean(train_loss_list)
    print('train epoch loss: ', train_epoch_loss)
    return train_epoch_loss

  
  def valid(epoch):
      #pass
      model.eval() #https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
      with torch.no_grad():
        valid_loss_list = []
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            #print('inputs shape={} type={}'.format(inputs.shape, type(inputs)))
            outputs = model(inputs.cuda()) #forward pass
            loss = criterion(outputs, targets.cuda())
            valid_loss_list.append(loss.item())
        valid_epoch_loss = np.mean(valid_loss_list)
        print('epoch: %d valid_loss= %.5f' % (epoch, valid_epoch_loss))
      return valid_epoch_loss

  def normal_test():
      # pass
      model.eval()
      correct = 0
      for batch_idx, (inputs, targets) in enumerate(test_loader):
          inputs, targets = inputs.cuda(), targets.cuda()
          outputs = model(inputs)
          _, predicted = torch.max(outputs.data, 1)
          correct += predicted.eq(targets.data).cpu().sum().item()
      acc = 100. * correct / len(test_dataset)
      print('%s------------------------------------------------------ '
                'Precision@1: %.2f%% \n' % (datetime.now(), acc))
      summary_writer.add_scalar('Precision@1', acc, global_step=epoch)

  def test():
    # pass
    model.eval()
    correct = 0
    y_pred = []
    y_true = []
    max_list = []
    min_list = []
    with torch.no_grad():
        print('looping on test data for {} iterations'.format(len(test_loader)))
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            print('iteration: {}, inputs type: {}, targets type: {}'.format(batch_idx, inputs.dtype, targets.dtype))
            white_count = 0 #count of white pixel in an image
            inputs, targets = inputs.cuda(), targets.cuda()
            image0 = transforms.ToPILImage()(inputs[0]).convert('RGB')
            #image0 = image0.resize(size=(image0.size[0]-6, image0.size[1]-6))
            img = np.array(image0)
            #print('img dtype: ', img.dtype)
            new_img = np.zeros(shape=(img.shape), dtype='uint8')
            #print('new img dtype: ', new_img.dtype)
            #print('new img shape: ', new_img.shape)
            #inputs_np = inputs.cpu().numpy()
            #img = inputs_np[0]
            #img = np.moveaxis(img, 0, 2) #change to channel last
            #loop over each pixel
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    #print('img[{},{}], pixel={}'.format(i, j, img[i,j]))
                    if(img[i,j, 0] >= 200 and img[i,j, 1] >= 200 and img[i,j, 2] >= 200):
                        #print('img[{},{}], pixel={}'.format(i, j, img[i,j]))
                        #print('maybe white pixel')
                        white_count += 1
                    else:
                        new_img[i, j] = img[i, j]
            outputs = model(inputs)
            #print('output shape: ', outputs.shape)
            _, predicted = torch.max(outputs.data, 1)
            #print('predicted (y_pred[i]): ', predicted.item())
            y_pred.append(predicted.item())
            y_true.append(targets.item())
            #print('y_pred: ', y_pred)
            correct += predicted.eq(targets.data).cpu().sum().item()
            if(batch_idx % 1000 == 0):
                print('============ iteration # ', batch_idx, ' ===================')
                print('test inputs shape:', inputs.shape, ' targets shape: ', targets.shape)
                print('image0 size: ', image0.size, ' np image shape: ', img.shape, 
                        ' new image shape: ', new_img.shape)
                print('output shape: ', outputs.shape, ' type: ', type(outputs))
                print('predicted (y_pred[i]): ', predicted.item())
                print('targets (y_true[i]): ', targets.item())
                print('white pixels count: {}. total pixel count = {}x{}={}'.format(white_count, img.shape[0],img.shape[1], img.shape[0]*img.shape[1]))
                print('img max: ', np.amax(img), ' new img max: ', np.amax(new_img))
                print('img min: ', np.min(img), ' new img min: ', np.min(new_img))
                orig_img = Image.fromarray(img).resize(size=(256, 256))
                trans_img = Image.fromarray(new_img).resize(size=(256, 256))
                orig_img_name = os.path.join(out_path, 'exp{}_org_img_{}_class{}.png'.format(cfg.exp_id, 
                    batch_idx, targets.item()))
                trans_img_name = os.path.join(out_path, 'exp{}_trans_img_{}_class{}.png'.format(cfg.exp_id, 
                    batch_idx, targets.item()))
                orig_img.save(orig_img_name)
                trans_img.save(trans_img_name)
                print('img saved: ', orig_img_name)
                print('img saved: ', trans_img_name)

        acc = 100. * correct / len(test_dataset)
        print('%s------------------------------------------------------ \n'
          'Test Precision@1: %.2f%% \n' % (datetime.now(), acc))
        summary_writer.add_scalar('Precision@1', acc)
    print('y pred list type: ', type(y_pred))
    print('y true list type: ', type(y_true))
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    print('y pred shape: ', y_pred.shape)
    print('y true shape: ', y_true.shape)
    return acc, y_pred, y_true

  start_training = time.time()
  train_losses = []
  valid_losses = []
  for epoch in range(cfg.max_epochs):
    print('\nEpoch: %d =============' % epoch)
    train_loss = train(epoch)
    valid_loss = valid(epoch) #get loss value
    lr_schedu.step(epoch)
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
  from sklearn.metrics import confusion_matrix
  cm = confusion_matrix(y_pred=y_pred, y_true=y_true)
  print('confusion matrix')
  print(cm)
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
