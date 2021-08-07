from .COVID_data import *

datasets = {
    'covid_19_seg': COVID_seg_Dataset,
}


def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
