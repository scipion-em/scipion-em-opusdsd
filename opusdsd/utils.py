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
from pyworkflow.utils.process import runJob
from opusdsd import Plugin
import numpy as np
import mrcfile as mrc

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

def window_r(inputMask):
    mask = mrc.read(os.path.abspath(inputMask))

    in_vol_nonzeros = np.stack(np.nonzero(mask), axis=1)
    in_vol_mins = in_vol_nonzeros.min(axis=0)
    in_vol_maxs = in_vol_nonzeros.max(axis=0)
    in_vol_maxs = mask.shape[-1] - in_vol_maxs
    in_vol_min = min(in_vol_maxs.min(), in_vol_mins.min())
    mask_frac = (mask.shape[-1] - in_vol_min * 2 + 4) / mask.shape[-1]

    return min(mask_frac, 0.9)