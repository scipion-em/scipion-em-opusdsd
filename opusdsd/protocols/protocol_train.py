# **************************************************************************
# *
# * Authors:     Grigory Sharov (gsharov@mrc-lmb.cam.ac.uk)
# *
# * MRC Laboratory of Molecular Biology (MRC-LMB)
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

import pyworkflow.utils as pwutils
import pyworkflow.object as pwobj
from pyworkflow.constants import PROD
import pyworkflow.protocol.params as params

from .protocol_base import OpusDsdProtBase


class OpusDsdProtTrain(OpusDsdProtBase):
    """
    Protocol to train OPUS-DSD neural network.
    """
    _label = 'training VAE'
    _devStatus = PROD

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        OpusDsdProtBase._defineParams(self, form)
        form.getParam('numberOfThreads').default = pwobj.Integer(1)

    def _defineAdvancedParams(self, form):
        form.addSection(label='Advanced')
        group = form.addGroup('Encoder')
        group.addParam('qLayers', params.IntParam, default=3,
                       label='Number of hidden layers')
        group.addParam('qDim', params.IntParam, default=1024,
                       label='Number of nodes in hidden layers')

        group = form.addGroup('Decoder')
        group.addParam('pLayers', params.IntParam, default=3,
                       label='Number of hidden layers')
        group.addParam('pDim', params.IntParam, default=1024,
                       label='Number of nodes in hidden layers')

        form.addParam('extraParams', params.StringParam, default="",
                      label="Extra params",
                      help="Here you can provide all extra command-line "
                           "parameters. See *cryodrgn train_vae -h* for help.")

    # --------------------------- STEPS functions -----------------------------
    def runTrainingStep(self):
        # Create output folder
        pwutils.cleanPath(self.getOutputDir())
        pwutils.makePath(self.getOutputDir())

        # Call OPUS-DSD with the appropriate parameters
        self._runProgram('train_vae', self._getTrainingArgs())

    # --------------------------- INFO functions ------------------------------
    def _summary(self):
        summary = ["Training VAE for %d epochs." % self.numEpochs]

        return summary

    def _validate(self):
        errors = OpusDsdProtBase._validateBase(self)

        if self.inputParticles.get().poses is None:
            errors.append("Input particles have no poses (alignment)!")

        return errors

    # --------------------------- UTILS functions -----------------------------
    def _getTrainingArgs(self):
        parts = self.inputParticles.get()

        args = [
            parts.filename.get(),
            '--poses %s' % parts.poses,
            '--ctf %s' % parts.ctfs,
            '--zdim %d' % self.zDim,
            '-o %s ' % self.getOutputDir(),
            '-n %d' % self.numEpochs,
            '--preprocessed',
            '--max-threads %d ' % self.numberOfThreads,
            '--enc-layers %d' % self.qLayers,
            '--enc-dim %d' % self.qDim,
            '--dec-layers %d' % self.pLayers,
            '--dec-dim %d' % self.pDim
        ]

        if len(self.getGpuList()) > 1:
            args.append('--multigpu')

        if self.extraParams.hasValue():
            args.append(self.extraParams.get())

        return args
