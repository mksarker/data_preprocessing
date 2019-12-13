import unittest
import torch
import torch.nn as nn
# from mobilenet import  MobileNetV2
from models import my_models


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TestResNetWrapper(unittest.TestCase):

    def test_resnet_wrapper(self):
        net = my_models(num_classes=10)

        # print(net)
        params = list(net.parameters()) 
        count=count_parameters(net)
        print (count)

        x = torch.rand(2, 3, 32, 32)
        x = torch.autograd.Variable(x)
        x = net(x)
        # print(x.size())
        self.assertTrue(x is not None)

if __name__ == '__main__':
    unittest.main()
