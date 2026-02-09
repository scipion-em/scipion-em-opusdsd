# **************************************************************************
# *
# * Authors:     Grigory Sharov (gsharov@mrc-lmb.cam.ac.uk) [1]
# *              James Krieger (jmkrieger@cnb.csic.es) [2]
# *              Eduardo García (eduardo.garcia@cnb.csic.es) [2]
# *
# * [1] MRC Laboratory of Molecular Biology (MRC-LMB)
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
import os
import opusdsd
from pyworkflow.utils.process import runJob
from opusdsd import Plugin
from opusdsd.constants import *
import numpy as np
import mrcfile as mrc
import pickle

def generateVolumes(zValues, weights, config, outdir, Apix, boxSize, crop_vol_size, wr, downFrac, zDim):
    """ Call OPUS-DSD with the appropriate parameters to generate volumes """

    args = '--load %s ' % weights
    args += '--config %s ' % config
    args += '-o %s ' % os.path.abspath(outdir)
    args += '--prefix vol_ '

    np.savetxt(f'{outdir}/zfile.txt', zValues)
    zFile = os.path.abspath(os.path.join(outdir, 'zfile.txt'))
    args += '--zfile %s ' % zFile

    render_size = (int(float(boxSize) * float(downFrac)) // 2) * 2
    newApix = (Apix * boxSize / render_size) * float(crop_vol_size) / (float(boxSize) * float(downFrac) * wr)
    args += '--Apix %f ' % round(newApix, 2)

    args += '--zdim %d ' % int(zDim)

    runJob(None, Plugin.getProgram('eval_vol', gpus='0'), ''.join(args),
           env=Plugin.getEnviron())

def checkCropSize(boxSize, downFrac, crop_vol_size, trainApix):
    """ Check Opus-DSD Network crop_vol_size parameter and its dependency with candidateApix """

    candidates = trainApix + np.linspace(-1, 1, 10000)
    candidates = candidates[np.argsort(np.abs(candidates - trainApix))]
    best_apix = trainApix
    found = False
    window_r = crop_vol_size / (float(boxSize) * float(downFrac))
    for candidate in candidates:
        ratio = trainApix / candidate
        render_size = (int(float(boxSize) * float(downFrac) * ratio + 1e-6) // 2) * 2
        final_size = int(render_size * window_r) // 2 * 2
        if final_size == crop_vol_size:
            best_apix = candidate
            found = True
            break

    if not found: print(f'WARNING: No exact match found for size {crop_vol_size}. Using original.')
    return best_apix

def getAnnotateSpaceArguments(particles, gpu_id=None):
    server_functions_path = os.path.join(os.path.dirname(opusdsd.__file__), "utils", "annotate_space_server.py")
    args = (f"--config {particles.getFlexInfo().getAttr(CONFIG)} --load {particles.getFlexInfo().getAttr(WEIGHTSNEW)}"
            f" --server_functions_path {server_functions_path} --env_name {opusdsd.Plugin.getOpusDsdEnvActivation().split(' ')[-1]}")

    if gpu_id is not None:
        args += f" --gpu_id {gpu_id}"

    return args