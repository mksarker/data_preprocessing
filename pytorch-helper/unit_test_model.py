import unittest
import torch
import torch.nn as nn
from emanet import EMANet
from ptflops import get_model_complexity_info
from torchvision import models
from efficientnet_pytorch import EfficientNet

from pytorch_memlab import MemReporter


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TestWrapper(unittest.TestCase):

    def test_wrapper(self):
        net = EfficientNet.from_pretrained("efficientnet-b3")

        # print(net)
        # params = list(net.parameters()) 
        # count=count_parameters(net)
        # print ("Params:",count/1000000)

        ## params and MACs
        macs, params = get_model_complexity_info(net, (3, 244, 244), as_strings=True,
                                        print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        ### Input for check model
        x = torch.rand(2, 3, 224, 224)
        x = torch.autograd.Variable(x)
       
        ## model size
        mem_params = sum([param.nelement()*param.element_size() for param in net.parameters()])
        mem_bufs = sum([buf.nelement()*buf.element_size() for buf in net.buffers()])
        mem = mem_params + mem_bufs # in bytes
        print('Memory Size:', mem/1000000)
        ## Memory 
        reporter = MemReporter(net)
        reporter.report()

        ### model implementation
        x = net(x)

        self.assertTrue(x is not None)

if __name__ == '__main__':
    unittest.main()
