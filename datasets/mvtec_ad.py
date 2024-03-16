import math
import pickle
from collections import defaultdict

import json
import random
import os
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing, read_json, write_json

CLSNAMES = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
            'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
CLSNAMES_map_index = {}
for k, index in zip(CLSNAMES, range(len(CLSNAMES))):
    CLSNAMES_map_index[k] = index


class AdDatum(Datum):
    def __init__(self, impath="", label=0, domain=0, classname="",
                 mask_path="",
                 specie_name="",
                 reality_name=""):
        self.mask_path = mask_path
        self.specie_name = specie_name
        self.reality_name = reality_name
        super().__init__(impath, label, domain, classname)


@DATASET_REGISTRY.register()
class MVTecAD(DatasetBase):
    dataset_dir = "mvtec_ad"

    def __init__(self, cfg):
        self.root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'mvtec_anomaly_detection')
        self.split_path = os.path.join(self.dataset_dir, 'split_mvtec.json')
        self.meta_path = os.path.join(self.dataset_dir, 'meta.json')
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        # self.transform = transform
        # self.target_transform = target_transform

        self.data_all = []
        self.class_names = ['good product', 'anomaly product']
        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            meta_info = json.load(open(self.meta_path, 'r'))
            raw_trainval = meta_info['train']
            raw_test = meta_info['test']
            trainval = self.read_data(raw_trainval, self.class_names, self.image_dir)
            train, val = self.split_trainval(trainval)
            test = self.read_data(raw_test, self.class_names, self.image_dir)
            self.save_split(train, val, test, filepath=self.split_path, path_prefix=self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = self.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    def generate_fewshot_dataset(self, *data_sources, num_shots=-1):
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_reality(data_source)
            dataset = []

            for reality, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output

    @staticmethod
    def split_dataset_by_reality(data_source):
        output = defaultdict(list)

        for item in data_source:
            output[item.reality_name].append(item)

        return output

    @staticmethod
    def read_data(meta_info, class_names, image_dir):
        '''
        meta_info = meta['train'] / meta['val']
        meta_info = {
            "cls_name": [
                "img_path":"",
                "cls_name":
                ...
            ]
        }
        aditems = [
            [
                impath,
                label,
                cls_name,
            ]
            ...
        ]
        '''
        items = []
        reality_names = list(meta_info.keys())
        for reality_name in reality_names:
            lines = meta_info[reality_name]
            for line in lines:
                impath = os.path.join(image_dir, line['img_path'])
                mask_path = line['mask_path']
                # classname = line['cls_name']
                classname = class_names[line['anomaly']]
                specie_name = line['specie_name']
                label = line['anomaly']
                item = AdDatum(impath=impath, label=label, classname=classname,
                               mask_path=mask_path, specie_name=specie_name, reality_name=reality_name)
                items.append(item)
        return items

    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        p_trn = 1 - p_val
        print(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)

        return train, val

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                mask_path = item.mask_path
                specie_name = item.specie_name
                reality_name = item.reality_name
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname, mask_path, specie_name, reality_name))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname, mask_path, specie_name, reality_name in items:
                impath = os.path.join(path_prefix, impath)
                item = AdDatum(impath=impath, label=int(label), classname=classname,
                               mask_path=mask_path,
                               specie_name=specie_name,
                               reality_name=reality_name)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args

        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        else:
            selected = labels[m:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}

        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = AdDatum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname,
                    mask_path=item.mask_path,
                    reality_name=item.reality_name,
                    specie_name=item.specie_name
                )
                dataset_new.append(item_new)
            output.append(dataset_new)

        return output
