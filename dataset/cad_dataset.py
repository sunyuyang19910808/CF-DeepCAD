from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
import h5py
import random
import warnings
import numpy as np
from cadlib.macro import *


def get_dataloader(phase, config, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle

    dataset = CADDataset(phase, config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers,
                            worker_init_fn=np.random.seed())
    return dataloader


class CADDataset(Dataset):
    def __init__(self, phase, config):
        super(CADDataset, self).__init__()
        self.raw_data = os.path.join(config.data_root, "cad_vec") # h5 data root
        self.phase = phase
        self.aug = config.augment
        self.path = os.path.join(config.data_root, "train_val_test_split.json")
        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]

        self.max_n_loops = config.max_n_loops          # Number of paths (N_P)
        self.max_n_curves = config.max_n_curves            # Number of commands (N_C)
        self.max_total_len = config.max_total_len
        self.size = 256
        self._warned_bad_samples = set()

    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)

    def _load_cad_vec(self, index):
        last_error = None
        total = len(self.all_data)
        for offset in range(total):
            candidate_index = (index + offset) % total
            data_id = self.all_data[candidate_index]
            h5_path = os.path.join(self.raw_data, data_id + ".h5")
            try:
                with h5py.File(h5_path, "r") as fp:
                    cad_vec = fp["vec"][:] # (len, 1 + N_ARGS)
                if cad_vec.shape[0] > self.max_total_len:
                    raise ValueError(
                        "cad_vec length {} exceeds max_total_len {}.".format(
                            cad_vec.shape[0], self.max_total_len
                        )
                    )
                if offset > 0:
                    warnings.warn(
                        "Sample {} is unavailable; falling back to {}.".format(
                            self.all_data[index], data_id
                        ),
                        RuntimeWarning,
                    )
                return data_id, cad_vec
            except (OSError, FileNotFoundError, KeyError, ValueError) as exc:
                last_error = exc
                if data_id not in self._warned_bad_samples:
                    warnings.warn(
                        "Skipping invalid cad_vec sample {}: {}".format(data_id, exc),
                        RuntimeWarning,
                    )
                    self._warned_bad_samples.add(data_id)
                continue
        raise RuntimeError("No valid cad_vec samples could be loaded for phase {}.".format(self.phase)) from last_error

    def __getitem__(self, index):
        data_id, cad_vec = self._load_cad_vec(index)

        if self.aug and self.phase == "train":
            command1 = cad_vec[:, 0]
            ext_indices1 = np.where(command1 == EXT_IDX)[0]
            # if len(ext_indices1) > 1 and random.randint(0, 1) == 1:
            if len(ext_indices1) > 1 and random.uniform(0, 1) > 0.5:
                ext_vec1 = np.split(cad_vec, ext_indices1 + 1, axis=0)[:-1]
        
                data_id2 = self.all_data[random.randint(0, len(self.all_data) - 1)]
                h5_path2 = os.path.join(self.raw_data, data_id2 + ".h5")
                with h5py.File(h5_path2, "r") as fp:
                    cad_vec2 = fp["vec"][:]
                command2 = cad_vec2[:, 0]
                ext_indices2 = np.where(command2 == EXT_IDX)[0]
                ext_vec2 = np.split(cad_vec2, ext_indices2 + 1, axis=0)[:-1]
        
                n_replace = random.randint(1, min(len(ext_vec1) - 1, len(ext_vec2)))
                old_idx = sorted(random.sample(list(range(len(ext_vec1))), n_replace))
                new_idx = sorted(random.sample(list(range(len(ext_vec2))), n_replace))
                for i in range(len(old_idx)):
                    ext_vec1[old_idx[i]] = ext_vec2[new_idx[i]]
        
                sum_len = 0
                new_vec = []
                for i in range(len(ext_vec1)):
                    sum_len += len(ext_vec1[i])
                    if sum_len > self.max_total_len:
                        break
                    new_vec.append(ext_vec1[i])
                cad_vec = np.concatenate(new_vec, axis=0)

        pad_len = self.max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        command = cad_vec[:, 0]
        args = cad_vec[:, 1:]
        command = torch.tensor(command, dtype=torch.long)
        args = torch.tensor(args, dtype=torch.long)
        return {"command": command, "args": args, "id": data_id}

    def __len__(self):
        return len(self.all_data)
