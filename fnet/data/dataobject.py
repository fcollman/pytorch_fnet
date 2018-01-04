class DataObject(object):
    def __init__(self):
        pass

    def get_size(self, dim_sel):
        pass

    def get_scales(self):
        d={
            'x':1,
            'y':1,
            'z':1,
            't':1
        }
        return d

    def get_volume(self, **selection):
        pass
    
