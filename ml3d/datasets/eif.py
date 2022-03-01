import open3d as o3d
import numpy as np
import os, glob, pickle
from os.path import join, exists, dirname, abspath, isdir
from .base_dataset import BaseDataset, BaseDatasetSplit
from pathlib import Path
import logging
import re
import random

log = logging.getLogger(__name__)


class EIF(BaseDataset):
    def __init__(
        self,
        bucket,
        dataset_path,
        name="EIF",
        task="segmentation",
        use_cache=False,
        test_result_folder="./test",
        cache_dir="./logs/cache",
        # TODO: num_points=TODO
        **kwargs
    ):
        super().__init__(
            dataset_path=dataset_path,
            name=name,
            task=task,
            cache_dir=cache_dir,
            use_cache=use_cache,
            test_result_folder=test_result_folder,
            # num_points=num_points, TODO
            **kwargs
        )

        self.bucket = bucket
        self.all_files = [
            blob.name
            for blob in bucket.list_blobs(prefix=dataset_path)
            if ".xyz" in blob.name
        ]

        assert (len(self.all_files) > 0), f"Invalid dataset path: {dataset_path}"
        self.num_files = len(self.all_files)

    @staticmethod
    def get_label_to_names():
        label_to_names = {
            0: "BlindFlange",
            1: "Cross",
            2: "Elbow 90",
            3: "Elbow non 90",
            4: "Flange",
            5: "Flange WN",
            6: "Olet",
            7: "OrificeFlange",
            8: "Pipe",
            9: "Reducer CONC",
            10: "Reducer ECC",
            11: "Reducer Insert",
            12: "Safety Valve",
            13: "Strainer",
            14: "Tee",
            15: "Tee RED",
            16: "Valve",
        }
        return label_to_names

    @staticmethod
    def get_name_to_label():
        label_to_names = {
            "BlindFlange": 0,
            "Cross": 1,
            "Elbow 90": 2,
            "Elbow non 90": 3,
            "Flange": 4,
            "Flange WN": 5,
            "Olet": 6,
            "OrificeFlange": 7,
            "Pipe": 8,
            "Reducer CONC": 9,
            "Reducer ECC": 10,
            "Reducer Insert": 11,
            "Safety Valve": 12,
            "Strainer": 13,
            "Tee": 14,
            "Tee RED": 15,
            "Valve": 16,
        }
        return label_to_names

    def get_split(self, split):
        return EIFSplit(self, split=split)

    def get_split_list(self, split):
        random.seed(100)  # ensure consistent splitting
        all_files_shuffled = random.sample(self.all_files, len(self.all_files))
        training_percentage = 0.9  # TODO may want to change this
        split_index = int(training_percentage * len(self.all_files))

        if split in ["test", "testing", "val", "validation"]:
            file_list = all_files_shuffled[split_index:-1]
        elif split in ["train", "training"]:
            file_list = all_files_shuffled[:split_index]
        elif split in ["all"]:
            file_list = self.all_files
        else:
            raise ValueError("Invalid split {}".format(split))

        return file_list

    def is_tested(self, attr):
        # checks whether attr['name'] is already tested.
        return

    def save_test_result(self, results, attr):
        # save results['predict_labels'] to file.
        return


class EIFSplit(BaseDatasetSplit):
    def __init__(self, dataset, split="train"):
        super().__init__(dataset, split=split)
        self.bucket = dataset.bucket
        log.info("Found {} pointclouds for {}".format(len(self.path_list), split))

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        point_cloud_path = self.path_list[idx]
        pc_blob = self.bucket.get_blob(point_cloud_path)

        # TODO: This procedure may be inefficient. Consider changing this if speed becomes an issue
        f = open('tmp/tmp_pc.xyz', 'wb')
        data = pc_blob.download_to_file(f)
        f.close()
        point_cloud = o3d.io.read_point_cloud('tmp/tmp_pc.xyz', format="xyz")

        label_name = self.get_attr(idx)["label"]
        label = self.dataset.get_name_to_label()[label_name]

        # All points in each point cloud are the same label
        # for the EIF dataset, hence this line
        labels = np.array(
            [label for i in range(len(point_cloud.points))], dtype=np.int32
        )

        point_cloud_data = {
            "point": np.asarray(point_cloud.points, dtype=np.float32),
            "label": labels,
        }
        return point_cloud_data

    def get_attr(self, idx):
        path = self.path_list[idx]
        name_with_filetype = path.split("/")[-1]
        name = name_with_filetype.split(".")[0]
        label = path.split("/")[-2]
        return {"name": name, "path": path, "split": self.split, "label": label}
