import zarr, os, warnings
# import s3fs
from OME_Zarr.src.core.OMEZarrCore import Collection, ZarrayManipulations
from OME_Zarr.src.core import utils

class BaseReader(Collection, ZarrayManipulations): # TODO: Add s3fs stuff here.
    def __init__(self,
                 basepath: str
                 ):
        Collection.__init__(self, basepath)
        if self.is_multiscales:
            ZarrayManipulations.__init__(self, self.base)

class _Labels:
    def __init__(self):
        self.label_names = [item for item in self.group_keys if 'multiscales' in self.base[item].attrs]
        self.image_labels = {}
        self.image_label_paths = {}
        for pth in self.label_names:
            image_label_path = os.path.join(self.root_path, pth)
            self.image_label_paths[pth] = image_label_path
            self.image_labels[pth] = OMEZarrObject(image_label_path)

class _MultiSeries:
    def __init__(self):
        self.image_names = [item for item in self.group_keys if 'multiscales' in self.base[item].attrs]
        self.images = {}
        self.image_paths = {}
        for pth in self.image_names:
            image_path = os.path.join(self.root_path, pth)
            self.image_paths[pth] = image_path
            self.images[pth] = OMEZarrObject(image_path)

class OMEZarrObject(BaseReader, _MultiSeries, _Labels):
    def __init__(self,
                 basepath: str
                 ):
        """ The main reader class """
        BaseReader.__init__(self, basepath)
        if self.is_labels:
            _Labels.__init__(self)
        elif self.is_multiseries:
            _MultiSeries.__init__(self)
        elif self.is_multiscales:
            pass
        elif self.is_unstructured:
            warnings.warn(f'The data at this path does not follow specifications: \n {basepath}')
        else:
            raise Exception(f'The data at this path cannot be resolved.\n: {basepath}\n This should be addressed.')
    def __getitem__(self, item):
        if self.is_labels:
            return self.image_labels[item]
        elif self.is_multiseries:
            return self.images[item]
        else:
            raise Exception('Data is neither multiseries nor labels.')

class OMEZarr:
    def __init__(self,
                 directory: str
                 ):
        """ A class that classifies OME-Zarr data as image and label objects. """
        self.baseobj = None
        self.labelobj = None
        self.unstructured_collections = {}
        self.collection_paths, self.multiscales_paths, self.array_paths = utils.get_collection_paths(directory, return_all = True)
        for fpath in self.collection_paths:
            obj = OMEZarrObject(fpath)
            if fpath == directory:
                self.baseobj = obj
            elif obj.is_labels:
                self.labelobj = obj
            else:
                self.unstructured_collections[os.path.basename(fpath)] = obj

