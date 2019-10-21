"""
Dataset classes for common uses
"""
import random

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Base Dataset

    Args:
        base_dir:
            dataset directory
    """
    def __init__(self, base_dir):
        self._base_dir = base_dir
        self.aux_attrib = {}
        self.aux_attrib_args = {}
        self.ids = []  # must be overloaded in subclass

    def add_attrib(self, key, func, func_args):
        """
        Add attribute to the data sample dict

        Args:
            key:
                key in the data sample dict for the new attribute
                e.g. sample['click_map'], sample['depth_map']
            func:
                function to process a data sample and create an attribute (e.g. user clicks)
            func_args:
                extra arguments to pass, expected a dict
        """
        if key in self.aux_attrib:
            raise KeyError("Attribute '{0}' already exists, please use 'set_attrib'.".format(key))
        else:
            self.set_attrib(key, func, func_args)

    def set_attrib(self, key, func, func_args):
        """
        Set attribute in the data sample dict

        Args:
            key:
                key in the data sample dict for the new attribute
                e.g. sample['click_map'], sample['depth_map']
            func:
                function to process a data sample and create an attribute (e.g. user clicks)
            func_args:
                extra arguments to pass, expected a dict
        """
        self.aux_attrib[key] = func
        self.aux_attrib_args[key] = func_args

    def del_attrib(self, key):
        """
        Remove attribute in the data sample dict

        Args:
            key:
                key in the data sample dict
        """
        self.aux_attrib.pop(key)
        self.aux_attrib_args.pop(key)

    def subsets(self, sub_ids, sub_args_lst=None):
        """
        Create subsets by ids

        Args:
            sub_ids:
                a sequence of sequences, each sequence contains data ids for one subset
            sub_args_lst:
                a list of args for some subset-specific auxiliary attribute function
        """

        indices = [[self.ids.index(id_) for id_ in ids] for ids in sub_ids]
        if sub_args_lst is not None:
            subsets = [Subset(dataset=self, indices=index, sub_attrib_args=args)
                       for index, args in zip(indices, sub_args_lst)]
        else:
            subsets = [Subset(dataset=self, indices=index) for index in indices]
        return subsets

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class PairedDataset(Dataset):
    """
    Make pairs of data from dataset

    When 'same=True',
        a pair contains data from same datasets,
        and the choice of datasets for each pair is random.
        e.g. [[ds1_3, ds1_2], [ds3_1, ds3_2], [ds2_1, ds2_2], ...]
    When 'same=False',
            a pair contains data from different datasets,
            if 'n_elements' <= # of datasets, then we randomly choose a subset of datasets,
                then randomly choose a sample from each dataset in the subset
                e.g. [[ds1_3, ds2_1, ds3_1], [ds4_1, ds2_3, ds3_2], ...]
            if 'n_element' is a list of int, say [C_1, C_2, C_3, ..., C_k], we first
                randomly choose k(k < # of datasets) datasets, then draw C_1, C_2, ..., C_k samples
                from each dataset respectively.
                Note the total number of elements will be (C_1 + C_2 + ... + C_k).

    Args:
        datasets:
            source datasets, expect a list of Dataset
        n_elements:
            number of elements in a pair
        max_iters:
            number of pairs to be sampled
        same:
            whether data samples in a pair are from the same dataset or not,
            see a detailed explanation above.
        pair_based_transforms:
            some transformation performed on a pair basis, expect a list of functions,
            each function takes a pair sample and return a transformed one.
    """
    def __init__(self, datasets, n_elements, max_iters, same=True,
                 pair_based_transforms=None):
        super().__init__()
        self.datasets = datasets
        self.n_datasets = len(self.datasets)
        self.n_data = [len(dataset) for dataset in self.datasets]
        self.n_elements = n_elements
        self.max_iters = max_iters
        self.pair_based_transforms = pair_based_transforms
        if same:
            if isinstance(self.n_elements, int):
                datasets_indices = [random.randrange(self.n_datasets)
                                    for _ in range(self.max_iters)]
                self.indices = [[(dataset_idx, data_idx)
                                 for data_idx in random.choices(range(self.n_data[dataset_idx]),
                                                                k=self.n_elements)]
                                for dataset_idx in datasets_indices]
            else:
                raise ValueError("When 'same=true', 'n_element' should be an integer.")
        else:
            if isinstance(self.n_elements, list):
                self.indices = [[(dataset_idx, data_idx)
                                 for i, dataset_idx in enumerate(
                                     random.sample(range(self.n_datasets), k=len(self.n_elements)))
                                 for data_idx in random.sample(range(self.n_data[dataset_idx]),
                                                               k=self.n_elements[i])]
                                for i_iter in range(self.max_iters)]
            elif self.n_elements > self.n_datasets:
                raise ValueError("When 'same=False', 'n_element' should be no more than n_datasets")
            else:
                self.indices = [[(dataset_idx, random.randrange(self.n_data[dataset_idx]))
                                 for dataset_idx in random.sample(range(self.n_datasets),
                                                                  k=n_elements)]
                                for i in range(max_iters)]

    def __len__(self):
        return self.max_iters

    def __getitem__(self, idx):
        sample = [self.datasets[dataset_idx][data_idx]
                  for dataset_idx, data_idx in self.indices[idx]]
        if self.pair_based_transforms is not None:
            for transform, args in self.pair_based_transforms:
                sample = transform(sample, **args)
        return sample


class Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Args:
        dataset:
            The whole Dataset
        indices:
            Indices in the whole set selected for subset
        sub_attrib_args:
            Subset-specific arguments for attribute functions, expected a dict
    """
    def __init__(self, dataset, indices, sub_attrib_args=None):
        self.dataset = dataset
        self.indices = indices
        self.sub_attrib_args = sub_attrib_args

    def __getitem__(self, idx):
        if self.sub_attrib_args is not None:
            for key in self.sub_attrib_args:
                # Make sure the dataset already has the corresponding attributes
                # Here we only make the arguments subset dependent
                #   (i.e. pass different arguments for each subset)
                self.dataset.aux_attrib_args[key].update(self.sub_attrib_args[key])
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
