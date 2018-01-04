import pandas as pd
import warnings
import os
from fnet.data.czireader import CziReader
from fnet.transforms import Resizer
from fnet.data.dataset import CsvDataSet, get_str_transform

class CziDataSet(CsvDataSet):
    def __init__(self,
                 scale_z = 0.3,
                 scale_xy = 0.3,
                 *args,
                 **kwargs
    ):
        """Create dataset from train/test DataFrames.
        
        Parameters:
        df_train - pandas.DataFrame, where each row is a DataSet element
        df_test - pandas.DataFrame, same columns as above
        scale_z - desired um/px size for z-dimension
        scale_xy - desired um/px size for x, y dimensions
        transforms - list/tuple of transforms, where each element is a transform or transform list to be applied
                     to a component of a DataSet element
        """
        super(CziDataSet,self).__init__(*args,**kwargs)

        self.scale_z = scale_z
        self.scale_xy = scale_xy
        self._czi = None
        self._last_loaded = None

    def get_signal(self,idx):
        row = self._df_active.ilox(idx)
        d = {
            'chan':row.channel_signal,
            'time_slice':row.get('time_slice',None)
        }
        return d

    def get_target(self,idx):
        row = self._df_active.ilox(idx)
        d = {
            'chan':row.channel_target,
            'time_slice':row.get('time_slice',None)
        }
        return d

    def get_selection(self,idx,sel):
        if sel==0:
            return self.get_signal(idx)
        if sel==1:
            return self.get_target(idx)

    def get_data_object(self,idx):
        path = self._df_active['path_czi'].iloc[idx]
        if ('_last_loaded' not in vars(self)) or self._last_loaded != path:
            print('reading:', path)
            try:
                self._czi = CziReader(path)
                self._last_loaded = path
            except Exception as e:
                warnings.warn('could not read file: {}'.format(path))
                warnings.warn(str(e))
                return None

        return self._czi

    def get_resizer(self,dataobj):
        dict_scales = dataobj.get_scales()
        scales_orig = [dict_scales.get(dim) for dim in 'zyx']
        # print('pixel scales:', scales_orig)
        if self.scale_z is not None or self.scale_xy is not None:
            if None in scales_orig:
                warnings.warn('bad pixel scales in {:s} | scales: {:s}'.format(path, str(scales_orig)))
                return None
            scales_wanted = [self.scale_z, self.scale_xy, self.scale_xy]
            factors_resize = list(map(lambda a, b : a/b if None not in (a, b) else 1.0, scales_orig, scales_wanted))
            # print('factors_resize:', factors_resize)
            resizer = Resizer(factors_resize)
        else:
            resizer = None
        return resizer
      
    def is_timelapse(self):
        return 'time_slice' in self._df_active.columns

    def __len__(self):
        return len(self._df_active)

    def get_name(self, idx, *args):
        return self._df_active['path_czi'].iloc[idx]

    def __repr__(self):
        return 'DataSet({:d} train elements, {:d} test elements)'.format(len(self.df_train), len(self.df_test))

    def __str__(self):
        
        if id(self.df_train) == id(self.df_test):
            n_unique = self.df_train.shape[0]
        else:
            n_unique = self.df_train.shape[0] + self.df_test.shape[0]
        str_active = 'train' if self._train_select else 'test'
        str_list = []
        str_list.append('{}:'.format(self.__class__.__name__))
        str_list.append('active_set: ' + str_active)
        str_list.append('scale_z: ' + str(self.scale_z) + ' um/px')
        str_list.append('scale_xy: ' + str(self.scale_xy) + ' um/px')
        str_list.append('train/test/total: {:d}/{:d}/{:d}'.format(len(self.df_train),
                                                                  len(self.df_test),
                                                                  n_unique))
        str_list.append('transforms: ' + get_str_transform(self.transforms))
        return os.linesep.join(str_list)

    def save_dataset_as_json(self,path):
        pass

    @staticmethod
    def load_dataset_as_json(Class,path):
        with open(path,'r') as fp:
            d=json.load(fp)
        Class(d['path_train_csv'],d['path_test_csv'])
if __name__ == '__main__':
    raise NotImplementedError
