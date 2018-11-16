import torch
import pdb
from fnet.nn_modules.fnet_nn_3d import _Net_recurse

class Net(torch.nn.Module):
    def __init__(self,n_in_channels=3,n_out_channels=1,depth=2,mult_chan=2):
        super().__init__()
        mult_chan = mult_chan
        depth = depth
        self.net_recurse = _Net_recurse(n_in_channels=n_in_channels, mult_chan=mult_chan, depth=depth)
        self.conv_ext = torch.nn.Conv3d(mult_chan*n_in_channels, mult_chan, kernel_size = 5, padding=2)
        self.bn1 = torch.nn.BatchNorm3d(mult_chan)
        self.relu1 = torch.nn.ReLU()
        self.conv_out = torch.nn.Conv3d(mult_chan,  n_out_channels, kernel_size=3, padding=1)

        
    def forward(self, x):
        x_rec = self.net_recurse(x)
        x_rec_ext = self.conv_ext(x_rec)
        x_rec_n = self.bn1(x_rec_ext)
        x_relu = self.relu1(x_rec_n)
        return self.conv_out(x_relu)
