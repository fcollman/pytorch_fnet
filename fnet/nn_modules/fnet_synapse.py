import torch
from fnet.nn_modules.fnet_nn_3d import Net as FNet

class Net(FNet):
    def __init__(self,n_in_channels=4,n_out_channels=1,depth=2,mult_chan=2):
        super(Net,self).__init__(n_in_channels=n_in_channels,n_out_channels=n_out_channels,depth=depth,mult_chan=mult_chan)
