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
import pwem
import pyworkflow.utils as pwutils
from pyworkflow import Config

from .constants import *

__version__ = '3.3.0'
_references = ['Luo2023']
_logo = "cryodrgn_logo.png"

class Plugin(pwem.Plugin):
    _url = "https://github.com/scipion-em/scipion-em-opusdsd"
    _supportedVersions = VERSIONS

    @classmethod
    def _defineVariables(cls):
        cls._defineVar(OPUSDSD_ENV_ACTIVATION, DEFAULT_ACTIVATION_CMD)
        cls._defineEmVar(OPUSDSD_HOME, getOpusDsdEnvName())

    @classmethod
    def getOpusDsdEnvActivation(cls):
        """ Remove the scipion home and activate the conda environment. """
        activation = cls.getVar(OPUSDSD_ENV_ACTIVATION)
        scipionHome = Config.SCIPION_HOME + os.path.sep

        return activation.replace(scipionHome, "", 1)

    @classmethod
    def getEnviron(cls):
        """ Setup the environment variables needed to launch cryoDRGN. """
        environ = pwutils.Environ(os.environ)
        if 'PYTHONPATH' in environ:
            # this is required for python virtual env to work
            del environ['PYTHONPATH']
        environ['LD_LIBRARY_PATH'] = ""
        return environ

    @classmethod
    def getDependencies(cls):
        """ Return a list of dependencies. Include conda if
        activation command was not found. """
        condaActivationCmd = cls.getCondaActivationCmd()
        neededProgs = []
        if not condaActivationCmd:
            neededProgs.append('conda')

        return neededProgs

    @classmethod
    def defineBinaries(cls, env):
        for ver in VERSIONS:
            cls.addOpusDsdPackage(env, ver,
                                  default=ver == OPUSDSD_DEFAULT_VER_NUM)

    @classmethod
    def addOpusDsdPackage(cls, env, version, default=False):
        ENV_NAME = getOpusDsdEnvName(version)
        FLAG = f"opusdsd_{version}_installed"

        if all(int(major) == 7 for major in CUDA_CAPABILITIES):
            FILE = 'environment.yml'
        else:
            FILE = 'environmentcu11torch11.yml'

        # try to get CONDA activation command
        installCmds = [
            cls.getCondaActivationCmd(),
            f'conda env create --name {ENV_NAME} --file {FILE} --yes &&',
            f'conda activate {ENV_NAME} &&',
            'pip install -e . &&',
        ]

        if all(int(major) == 7 for major in CUDA_CAPABILITIES):
            installCmds += [
                'pip install numpy==1.21.0 &&',
                'pip install seaborn==0.13.2 &&'
            ]
        else:
            installCmds += [
                'pip install pillow==10.4.0 &&'
            ]

        installCmds += [
            f'touch {FLAG}'  # Flag installation finished
        ]

        envPath = os.environ.get('PATH', "")
        # keep path since conda likely in there
        installEnvVars = {'PATH': envPath} if envPath else None

        branch = "main"
        url = "https://github.com/alncat/opusDSD"
        if os.path.exists(cls.getVar(OPUSDSD_HOME)):
            gitCmds = []
        else:
            gitCmds = [
                'cd .. &&',
                f'git clone -b {branch} {url} opusdsd-{version} &&',
                f'cd opusdsd-{version};'
            ]
        gitCmds.extend(installCmds)
        opusdsdCmds = [(" ".join(gitCmds), FLAG)]
        env.addPackage('opusdsd', version=version,
                       tar='void.tgz',
                       commands=opusdsdCmds,
                       neededProgs=cls.getDependencies(),
                       default=default,
                       vars=installEnvVars)

    @classmethod
    def getActivationCmd(cls):
        """ Return the activation command. """
        return '%s %s' % (cls.getCondaActivationCmd(),
                          cls.getOpusDsdEnvActivation())

    @classmethod
    def getProgram(cls, program, gpus='0', fromCryodrgn=True):
        """ Create Opus-DSD command line. """
        fullProgram = 'cd %s && %s && CUDA_VISIBLE_DEVICES=%s ' % (cls.getVar(OPUSDSD_HOME), cls.getActivationCmd(), gpus)
        if fromCryodrgn:
            fullProgram += 'python -m cryodrgn.commands.%s' % program
        else:
            fullProgram += 'sh %s/%s.sh' % (cls.getVar(OPUSDSD_HOME), program)
        return fullProgram

    @classmethod
    def getTorchLoadProgram(cls, workDir, file, output_file, option):
        """ Import Torch Load Program for Opus-DSD command line. """
        fullProgram = (
            f'cd {workDir} && {cls.getActivationCmd()} && '
            'python -c "'
            'import torch; '
            'import numpy as np; '
            f'data = torch.load(\'{file}\'); '
        )
        if option == 'pkl':
            fullProgram += (
                'npdata = {k: v.numpy() if hasattr(v, \'numpy\') else v for k, v in data.items()}; '
                f'np.savez(\'{output_file + ".npz"}\', **npdata); '
                f'np.savetxt(\'{output_file + ".txt"}\', npdata[list(npdata.keys())[0]])"'
            )
        elif option == 'weights':
            fullProgram += (
                'in_mask_param = data[\'model_state_dict\'][\'encoder.in_mask\']; '
                'in_mask_param = in_mask_param.squeeze(1); '
                'data[\'model_state_dict\'][\'encoder.in_mask\'] = in_mask_param; '
                f'torch.save(data, \'{output_file}\')"'
            )
        elif option == 'eval_vol':
            fullProgram += (
                'crop_vol_size = data[\'model_state_dict\'][\'encoder.in_mask\'].squeeze()[0]; '
                f'np.savetxt(\'{output_file + ".txt"}\', crop_vol_size.cpu().numpy())"'
            )
        return fullProgram

    @classmethod
    def getXmippProgram(cls, program):
        """ Import Xmipp Program. """
        fullProgram = 'cd %s && cd bin/ && ./%s' % (cls.getVar(XMIPP_HOME), program)
        return fullProgram

    @classmethod
    def getActiveVersion(cls, *args):
        """ Return the env name that is currently active. """
        envVar = cls.getVar(OPUSDSD_ENV_ACTIVATION)
        return envVar.split()[-1].split("-")[-1]