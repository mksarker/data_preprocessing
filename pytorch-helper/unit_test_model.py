import unittest
import torch
import torch.nn as nn
from emanet import EMANet
from ptflops import get_model_complexity_info
from torchvision import models

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TestResNetWrapper(unittest.TestCase):

    def test_resnet_wrapper(self):
        net = models.resnet50(num_classes=1000)

        # print(net)
        # params = list(net.parameters()) 
        # count=count_parameters(net)
        # print ("Params:",count/1000000)

        macs, params = get_model_complexity_info(net, (3, 244, 244), as_strings=True,
                                        print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        x = torch.rand(2, 3, 224, 224)
        x = torch.autograd.Variable(x)
        x = net(x)
        # print(x.size())
        self.assertTrue(x is not None)

if __name__ == '__main__':
    unittest.main()
