import os
import numpy as np
from fnet import get_vol_transformed
import pandas as pd
import pdb
import collections
import warnings
import tifffile
from fnet.data.transforms import Resizer
from .dataset import CsvDataSet, get_str_transform
from .dataobject import DataObject

class TiffObject(DataObject):
    def __init__(self,signal_filepath,target_filepath):
        self.signal_filepath = signal_filepath
        self.target_filepath = target_filepath
        self.signal_volume = tifffile.imread(self.signal_filepath)
        self.target_volume = tifffile.imread(self.target_filepath)

    def get_size(self, dim_sel):
        return self.signal_volume.shape[dim_sel]

    def get_scales(self):
        d={
            'x':1,
            'y':1,
            'z':1,
            't':1
        }
        return d

    def get_volume(self,sel=0):
        if sel==0:
            return self.signal_volume
        if sel==1:
            return self.target_volume

class ETFDataSet(CsvDataSet):
    def __init__(self,
                 scale_z = None,
                 scale_xy = None,
                 *args,
                 **kwargs
    ):
        """Create dataset from train/test DataFrames.
        
        Parameters:
        df_train - pandas.DataFrame, where each row is a DataSet element
        df_test - pandas.DataFrame, same columns as above
        transforms - list/tuple of transforms, where each element is a transform or transform list to be applied
                     to a component of a DataSet element
        """
        super(ETFDataSet,self).__init__(*args,**kwargs)
        # self.scale_z = scale_z
        # self.scale_xy = scale_xy
        self._train_select = True
        self._df_active = self.df_train
    
    def __len__(self):
        return len(self._df_active)
    def __repr__(self):
        return 'ETFDataSet({:d} train elements, {:d} test elements)'.format(len(self.df_train), len(self.df_test))

    def __str__(self):
        return "ETFDataSet"

    def use_train_set(self):
        self._train_select = True
        self._df_active = self.df_train
        
    def use_test_set(self):
        self._train_select = False
        self._df_active = self.df_test
    
    def get_selection(self,idx,sel):
        return {
            'sel':sel
            }

    def get_resizer(self,dataobj):
        return None

    def get_name(self, idx, *args):
        return self._df_active['path_signal'].iloc[idx]
   
    def get_data_object(self,idx):
        signal_path = self._df_active['path_signal'].iloc[idx]
        target_path = self._df_active['path_target'].iloc[idx]
        return TiffObject(signal_path,target_path)

if __name__ == '__main__':
    raise NotImplementedError
