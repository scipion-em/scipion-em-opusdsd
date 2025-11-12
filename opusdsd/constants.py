# **************************************************************************
# *
# * Authors:     Grigory Sharov (gsharov@mrc-lmb.cam.ac.uk) [1]
# *              James Krieger (jmkrieger@cnb.csic.es) [2]
# *              Eduardo García Delgado (eduardo.garcia@cnb.csic.es) [2]
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

import sys, subprocess

V0_3_2B = "0.3.2b"
V1_1_0 = "v1.1.0"

def getOpusDsdEnvName(version=V1_1_0):
    return "opusdsd-%s" % version

VERSIONS = [V0_3_2B, V1_1_0]
OPUSDSD_DEFAULT_VER_NUM = V1_1_0

OPUSDSD_HOME = 'OPUSDSD_HOME'
XMIPP_HOME = 'XMIPP_HOME'

DEFAULT_ENV_NAME = getOpusDsdEnvName(OPUSDSD_DEFAULT_VER_NUM)
DEFAULT_ACTIVATION_CMD = 'conda activate ' + DEFAULT_ENV_NAME
OPUSDSD_ENV_ACTIVATION = 'OPUSDSD_ENV_ACTIVATION'

KMEANS = 0
PCA = 1

VOLUME_SLICES = 0
VOLUME_CHIMERA = 1

# extra metadata attrs
Z_VALUES = "_opusdsdZValues"
WEIGHTS = "_opusdsdWeights"
WEIGHTSNEW = "_opusdsdWeightsNew"
CONFIG = "_opusdsdConfig"
ZDIM = "_opusdsdZDim"
DOWNFRAC = "_opusdsdDownFrac"
CROP_VOL_SIZE = "_opusdsdCropVolSize"
WINDOW_R = "_opusdsdWindowR"

# FlexHub program
OPUSDSD = "Opus-DSD"

# nvcc --version CUDA
#import re, subprocess
#command = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, check=True)
#match = re.search(r'release (\d+\.\d+)', command.stdout)
#CUDA_VERSION = float(match.group(1))

command = [
    'nvidia-smi',
    '--query-gpu=index,name,compute_cap',
    '--format=csv,noheader,nounits'
]
result = subprocess.run(command, capture_output=True, text=True, check=True)
compute_capabilities = result.stdout.strip().splitlines()

CUDA_CAPABILITIES = []
for line in compute_capabilities:
    CUDA_CAPABILITY = line.split(',')[2].strip().split('.')[0]
    CUDA_CAPABILITIES.append(CUDA_CAPABILITY)

# masks info
masks_info = [
    {"mask_name": "Masks/input_multimask_0.mrc", "rotate_relative_to": 2, "sigma_angles": 10, "sigma_offset": 2,
     "reference_name": "Masks/ref_0.mrc"},
    {"mask_name": "Masks/input_multimask_1.mrc", "rotate_relative_to": 1, "sigma_angles": 15, "sigma_offset": 3,
     "reference_name": "Masks/ref_1.mrc"},
    {"mask_name": "Masks/input_multimask_2.mrc", "rotate_relative_to": 1, "sigma_angles": 15, "sigma_offset": 3,
     "reference_name": "Masks/ref_2.mrc"},
    {"mask_name": "Masks/input_multimask_3.mrc", "rotate_relative_to": 1, "sigma_angles": 15, "sigma_offset": 3,
     "reference_name": "Masks/ref_3.mrc"},
]