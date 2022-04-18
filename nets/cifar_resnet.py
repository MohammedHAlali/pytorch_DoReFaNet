import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.quant_dorefa import *


class PreActBlock_conv_Q(nn.Module):
  '''Pre-activation version of the BasicBlock.'''
  #wbit = None
  #abit = None
  def __init__(self, wbit, abit, in_planes, out_planes, stride=1):
    super(PreActBlock_conv_Q, self).__init__()
    self.wbit = wbit
    self.abit = abit
    # remove quantizatio when bitwidth is 32
    if(self.wbit == 32 or self.abit == 32):
        #print('removing quantization functions when wbits or abits is 32')
        Conv2d = nn.Conv2d
        self.act_q = None
    else:
        Conv2d = conv2d_Q_fn(w_bit=wbit)
        self.act_q = activation_quantize_fn(a_bit=abit)

    self.bn0 = nn.BatchNorm2d(in_planes)
    self.conv0 = Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_planes)
    self.conv1 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

    self.skip_conv = None
    if stride != 1:
      self.skip_conv = Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
      self.skip_bn = nn.BatchNorm2d(out_planes)

  def forward(self, x):
    # remove quantization when wbit or abit is 32
    if(self.wbit == 32 or self.abit == 32):
      out = F.relu(self.bn0(x))
    else:
      out = self.act_q(F.relu(self.bn0(x)))

    if self.skip_conv is not None:
      shortcut = self.skip_conv(out)
      shortcut = self.skip_bn(shortcut)
    else:
      shortcut = x

    out = self.conv0(out)
    if(self.wbit == 32 or self.abit == 32):
      out = F.relu(self.bn1(out))
    else:
      out = self.act_q(F.relu(self.bn1(out)))
    out = self.conv1(out)
    out += shortcut
    return out


class PreActResNet(nn.Module):
  def __init__(self, block, num_units, wbit, abit, num_classes):
    super(PreActResNet, self).__init__()
    #self.dropout = dropout
    #first layer is not quantized, use bias = True
    self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)

    self.layers = nn.ModuleList()
    in_planes = 16
    strides = [1] * (num_units[0]) + \
              [2] + [1] * (num_units[1] - 1) + \
              [2] + [1] * (num_units[2] - 1)
    channels = [16] * num_units[0] + [32] * num_units[1] + [64] * num_units[2]
    for stride, channel in zip(strides, channels):
      self.layers.append(block(wbit, abit, in_planes, channel, stride))
      in_planes = channel

    self.bn = nn.BatchNorm2d(64)
    #if(self.dropout):
        # add Dropout
    #    self.dropout = nn.Dropout()
    self.logit = nn.Linear(64, num_classes)

  def forward(self, x):
    out = self.conv0(x)
    for layer in self.layers:
      out = layer(out)
    out = self.bn(out)
    #if(self.dropout):
    #    out = self.dropout(out)
    out = out.mean(dim=2).mean(dim=2)
    out = self.logit(out)
    return out


def resnet20(wbits, abits, num_classes):
  return PreActResNet(PreActBlock_conv_Q, [3, 3, 3], wbits, abits, num_classes)


def resnet56(wbits, abits, num_classes):
  return PreActResNet(PreActBlock_conv_Q, [9, 9, 9], wbits, abits, num_classes)


if __name__ == '__main__':
  features = []


  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)
    features.append(output.data.cpu().numpy())


  net = resnet20(wbits=1, abits=2)
  for m in net.modules():
    m.register_forward_hook(hook)

  y = net(torch.randn(1, 3, 32, 32))
  print(y.size())
