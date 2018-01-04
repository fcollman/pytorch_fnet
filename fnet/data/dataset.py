import os
import pickle
import numpy as np
from fnet import get_vol_transformed
import pandas as pd
import pdb
import collections
import warnings
from fnet.transforms import Resizer

def get_obj(a):
    a = a.replace('fnet.data', 'fnet.transforms') # hardcode for old naming
    if a is None:
        return None
    a_list = a.split('.')
    obj = getattr(sys.modules[__name__], a_list[0])
    for i in range(1, len(a_list)):
        obj = getattr(obj, a_list[i])
    return obj

def get_str_transform(transforms):
    # Return the string representation of the given transforms
    if transforms is None:
        return str(None)
    all_transforms = []
    for transform in transforms:
        if transform is None:
            all_transforms.append(str(None))
        elif isinstance(transform, (list, tuple)):
            str_list = []
            for t in transform:
                str_list.append(str(t))
            all_transforms.append(' => '.join(str_list))
        else:
            all_transforms.append(str(transform))
    return (os.linesep + '            ').join(all_transforms)

class DataSet(object):
    def __init__(self,transforms_source=None,transforms_target=None):
        self._train_select = True
        if transforms_source is not None:
            transforms_signal = map(get_obj,transforms_source)
        if transforms_target is not None:
            transforms_target = map(get_obj,transforms_target)
        self.transforms=(transforms_signal,transforms_target)
    
    def __len__(self):
        pass
    def __repr__(self):
        pass
    def __str__(self):
        pass

    def get_name(self, idx, *args):
        pass

    def use_train_set(self):
        pass
        
    def use_test_set(self):
        pass

    def get_selection(self,idx,sel):
        pass

    def get_name(self, idx, *args):
        pass
        
    def get_data_object(self,idx):
        pass

    def get_resizer(self,dataobj):
        return None
    def save_dataset_as_json(self,path):
        pass
    def load_dataset_as_json(self,path):
        pass
    def get_volume(self,dataobj,**selection):
        return dataobj.get_volume(**selection)

    def get_volumes(self,dataobj,idx,sels):
        volumes = []
        for i in range(len(sels)):
            d=self.get_selection(idx,sels[i])
            volume_pre=self.get_volume(dataobj,**d)
            volumes.append(volume_pre)
        return volumes

    def get_transforms(self,sel):
        transforms = []
        if self.transforms is not None:
            if isinstance(self.transforms[sel], collections.Iterable):
                transforms.extend(self.transforms[sel])
            else:
                transforms.append(self.transforms[sel])
        return transforms

    def apply_transforms_to_volumes(self,volumes,sels,resizer=None):
        tform_volumes = []
        for vol,sel in zip(volumes,sels):
            transforms = self.get_transforms(sel)
            if resizer is not None:
                transforms.append(resizer)
            tform_volumes.append(get_vol_transformed(vol, transforms))
        return tform_volumes

    def get_item_sel(self, idx, sel, apply_transforms=True):
            """Get item(s) from dataset element idx.

            idx - (int) dataset element index
            sel - (int or iterable) 0 for 'signal', 1 for 'target'
            """
            if isinstance(sel, int):
                assert sel >= 0
                sels = (sel, )
            elif isinstance(sel, collections.Iterable):
                sels = sel
            else:
                raise AttributeError
            dataobj = self.get_data_object(idx)
            resizer = self.get_resizer(dataobj)
            volumes = self.get_volumes(dataobj,idx,sels)
            if apply_transforms:
                volumes = self.apply_transforms_to_volumes(volumes,sels,resizer=resizer)
            return volumes[0] if isinstance(sel, int) else volumes

    def __getitem__(self, idx):
        """Returns arrays corresponding to files identified by file_tags in the folder specified by index.

        Once the files are read in as numpy arrays, apply the transformations specified in the constructor.

        Returns:
        volumes - n-element tuple or None. If the file read was successful, return tuple
                  of transformed arrays else return None
        """
        return self.get_item_sel(idx, (0, 1))

class CsvDataSet(DataSet):

    def __init__(self,path_train_csv=None,
                 path_test_csv=None,*args,**kwargs):
        super(CziDataSet,self).__init__(*args,**kwargs)
        self.df_train = pd.read_csv(path_train_csv) if path_train_csv is not None else pd.DataFrame()
        self.df_test = pd.read_csv(path_test_csv) if path_test_csv is not None else pd.DataFrame()
        self._df_active = self.df_train
        self._train_select = True
    
    def use_train_set(self):
        self._train_select = True
        self._df_active = self.df_train
        
    def use_test_set(self):
        self._train_select = False
        self._df_active = self.df_test