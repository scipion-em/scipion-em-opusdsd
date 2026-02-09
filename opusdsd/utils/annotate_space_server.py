# **************************************************************************
# *
# * Authors:     Eduardo García (eduardo.garcia@cnb.csic.es)     [2]
# *
# * [1] MRC Laboratory of Molecular Biology, MRC-LMB
# * [2] Unidad de  Biocomputacion, Centro Nacional de Biotecnologia, CSIC (CNB-CSIC)
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 3 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

import numpy as np
import torch
import torch.nn as nn
from cryodrgn.models import HetOnlyVAE
from cryodrgn.lattice import Lattice
from cryodrgn import utils

class HeterogeneityProgramInterface:
    def __init__(self, _path_template: str, _program_loading_params: dict):
        self.model = self.prepare_heterogeneity_program(**_program_loading_params)
        self.path_template = _path_template

    def prepare_heterogeneity_program(self, **kwargs) -> object:
        gpu_id = kwargs.pop("gpu_id", None)
        config = kwargs.pop("config", None)
        load = kwargs.pop("load", None)
        self.device = "cpu" if gpu_id is None else 'cuda:' + str(int(gpu_id))

        cfg = utils.load_pkl(config)

        in_dim = -1
        enc_mask = -1
        D = cfg['lattice_args']['D']
        zdim = cfg['model_args']['zdim']
        if "z_affine_dim" in cfg['model_args']:
            z_affine_dim = cfg['model_args']['z_affine_dim']
        else:
            z_affine_dim = 4
        lattice = Lattice(D, extent=0.5)
        downfrac = cfg['dataset_args']['downfrac']
        crop_vol_size = cfg['model_args']['down_vol_size']
        templateres = cfg['model_args']['templateres']
        window_r = crop_vol_size / ((D - 1) * downfrac)

        qlayers = cfg['model_args']['qlayers']
        qdim = cfg['model_args']['qdim']
        players = cfg['model_args']['players']
        pdim = cfg['model_args']['pdim']
        encode_mode = cfg['model_args']['encode_mode']
        pe_type = cfg['model_args']['pe_type']
        pe_dim = cfg['model_args']['pe_dim']
        domain = cfg['model_args']['domain']
        activation = cfg['model_args']['activation']
        self.Apix = cfg['model_args']['Apix']
        template_type = cfg['model_args']['template_type']

        activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[activation]
        model = HetOnlyVAE(lattice, qlayers, qdim, players, pdim,
                           in_dim, zdim, encode_mode=encode_mode, enc_mask=enc_mask,
                           enc_type=pe_type, enc_dim=pe_dim, domain=domain,
                           activation=activation, ref_vol=None, Apix=self.Apix,
                           template_type=template_type, device=self.device, ctf_grid=None, downfrac=1.0,
                           templateres=templateres, window_r=window_r, z_affine_dim=z_affine_dim)

        checkpoint = torch.load(load)
        print(checkpoint.keys())
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        pretrained_dict = checkpoint['encoder_state_dict']
        model_dict = model.encoder.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and "grid" not in k and "mask" not in k}
        model_dict.update(pretrained_dict)

        pretrained_dict = checkpoint['decoder_state_dict']
        if "ref_mask" in pretrained_dict:
            model.decoder.ref_mask = pretrained_dict["ref_mask"]
        model_dict = model.decoder.state_dict()
        for k in list(pretrained_dict.keys()):
            if k not in model_dict or pretrained_dict[k].shape != model_dict[k].shape:
                if k in model_dict:
                    print(k, pretrained_dict[k].shape, model_dict[k].shape)
                del pretrained_dict[k]
        model_dict.update(pretrained_dict)
        model.decoder.load_state_dict(model_dict)
        model = model.to(self.device)
        model.eval()
        return model

    def decode_state_from_latent(self, latent: np.array) -> None:
        latent = torch.from_numpy(latent.astype(np.float32)).to(self.device)
        for idx, zz in enumerate(latent):
            self.model.save_mrc(self.path_template.format(idx + 1).replace('.mrc', ''), enc=zz, Apix=self.Apix)