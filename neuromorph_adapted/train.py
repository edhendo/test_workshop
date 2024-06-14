# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from model.interpolation_net import *
from utils.arap import ArapInterpolationEnergy
from data.data import *
from utils.utils import k_fold_split_train_val_test, ParametersBase, getFiles
import sys

class NetParam(ParametersBase):
    def __init__(self):
        super().__init__()
        self.lr = 1e-4
        self.num_it = 75
        self.batch_size = 16
        self.num_timesteps = 0
        self.hidden_dim = 64
        self.lambd = 1
        self.lambd_geo = 50
        self.lambd_arap = 10

        self.log_freq = 10
        self.val_freq = 1

        self.log = True

def create_interpol(dataset, dataset_val=None, folder_weights_load=None, time_stamp=None, description="", param=NetParam(), hyp_param=None):
    if time_stamp is None:
        time_stamp = get_timestr()

    interpol_energy = ArapInterpolationEnergy()

    interpol_module = InterpolationModGeoEC(interpol_energy, param).to(device)

    preproc_mods = []
    preproc_mods.append(PreprocessRotateSame(dataset.axis))
    
    settings_module = timestep_settings(increase_thresh=30)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False)

    interpol = InterpolNet(interpol_module, train_loader, val_loader=val_loader, time_stamp=time_stamp, description=description, preproc_mods=preproc_mods, settings_module=settings_module)

    if folder_weights_load is not None:
        interpol.load_self(save_path(folder_str=folder_weights_load))

    interpol.i_epoch = 0

    return interpol


def train_main(fold_num=1, structure="brainstem"):
    data_folder = "/home/ed/correspondence/data/rigid_reg/zero_CoM_lung_cut/"

    # determine which meshes to use
    all_fnames = list(sorted(filter(lambda x: structure in x, getFiles(join(data_folder, "meshes/")))))
    train_inds, val_inds, _ = k_fold_split_train_val_test(dataset_size=len(all_fnames), fold_num=int(fold_num), seed=100457)
    train_fnames = [all_fnames[i].replace('.ply','') for i in train_inds]
    val_fnames = [all_fnames[i].replace('.ply','') for i in val_inds]

    if structure == "parotid" or structure == "submandibular":
        dataset_train = bilateral_dataset(data_folder, fnames=train_fnames, load_dist_mat=True)
        dataset_val = bilateral_dataset(data_folder, fnames=val_fnames, load_dist_mat=True)
    else:
        dataset_train = general_dataset(data_folder, fnames=train_fnames, load_dist_mat=True)
        dataset_val = general_dataset(data_folder, fnames=val_fnames, load_dist_mat=True)

    interpol = create_interpol(dataset_train, dataset_val=dataset_val, description=f"{structure}_fold{fold_num}", folder_weights_load=None)
    interpol.train()


if __name__ == "__main__":
    train_main(sys.argv[1], sys.argv[2])
