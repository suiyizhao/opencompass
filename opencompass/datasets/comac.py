import json
import os.path as osp

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class ComacDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        with open(osp.join(path, f'{name}.jsonl'), 'r') as f:
            data = json.load(f)
            # TODO: support multi-run conversations
            new_data = [{k: v
                         for k, v in d['conversation'][0].items()}
                        for d in data]
            return Dataset.from_list(new_data)
