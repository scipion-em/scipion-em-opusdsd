# **************************************************************************
# *
# * Authors:     Grigory Sharov (gsharov@mrc-lmb.cam.ac.uk) [1]
# *              James Krieger (jmkrieger@cnb.csic.es) [2]
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
import pyworkflow.utils as pwutils
import pyworkflow.object as pwobj
from pyworkflow.constants import PROD
import pyworkflow.protocol.params as params

from .protocol_base import OpusDsdProtBase

from pwem.objects import Volume, SetOfVolumes


class OpusDsdProtTrain(OpusDsdProtBase):
    """
    Protocol to train OPUS-DSD neural network.
    """
    _label = 'training CV'
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

        form.addParam('batchSize', params.IntParam, default=20, 
                      label='Batch size', expertLevel=params.LEVEL_ADVANCED,
                      help='Batch size for processing images')
        
        form.addParam('betaControl', params.FloatParam, default=1., 
                      label='Beta restraint strength for KL target',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Beta parameter for beta-VAE that controls the strength of '
                           'to control the strength of the standard Gaussian restraints')
        
        form.addParam('lamb', params.FloatParam, default=1.0, 
                      label='Restraint strength for umap prior',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='This controls the stretch of the UMAP-inspired prior for '
                            'the encoder network that encourages the encoding of structural '
                            'information for images in the same projection class')
        
        form.addParam('bfactor', params.FloatParam, default=4., 
                      label='B-factor for reconstruction',
                      expertLevel=params.LEVEL_ADVANCED)
        
        form.addParam('learningRate', params.FloatParam, default=1.2e-4, 
                      label='Learning rate', expertLevel=params.LEVEL_ADVANCED,
                      help='Learning rate in Adam optimizer')
        
        form.addParam('valFrac', params.FloatParam, default=0.2, 
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Validation image fraction',
                      help='fraction of images held for validation')
        
        form.addParam('downFrac', params.FloatParam, default=0.5, 
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Downsampling fraction',
                      help='downsample to this fraction of original size')
        
        form.addParam('templateres', params.IntParam, default=192, 
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Output size',
                      help='define the output size of 3d volume')
        
        form.addParam('useMask', params.BooleanParam, default=False,
                      expertLevel=params.LEVEL_ADVANCED,
                      label="Use a reference volume?")
        form.addParam('mask', params.PointerParam,
                      pointerClass='Mask',
                      expertLevel=params.LEVEL_ADVANCED,
                      condition='useMask',
                      label="Reference volume",
                      help="the solvent mask created from consensus model, our program will "
                            "focus on fitting the contents inside the mask (more specifically, "
                            "the 2D projection of a 3D mask). Since the majority part of image "
                            "doesn't contain electron density, using the original image size is "
                            "wasteful, by specifying a mask, our program will automatically "
                            "determine a suitable crop rate to keep only the region with densities.")

        form.addParam('extraParams', params.StringParam, default="",
                      label="Extra params",
                      help="Here you can provide all extra command-line "
                           "parameters. See *cryodrgn train_cv -h* for help.")

    # --------------------------- STEPS functions -----------------------------
    def runTrainingStep(self):
        # Call OPUS-DSD with the appropriate parameters
        self._runProgram('train_cv', self._getTrainingArgs())

    # --------------------------- INFO functions ------------------------------
    def _summary(self):
        summary = ["Training CV for %d epochs." % self.numEpochs]

        return summary

    # --------------------------- UTILS functions -----------------------------
    def _getTrainingArgs(self):
        args = [
            self._getFileName('input_parts'),
            '--poses %s' % self._getFileName('output_poses'),
            '--ctf %s' % self._getFileName('output_ctfs'),
            '--zdim %d' % self.zDim,
            '-o %s ' % self.getOutputDir(),
            '-n %d' % self.numEpochs,
            '--lazy-single',
            '--pe-type vanilla',
            '--encode-mode grad',
            '--template-type conv',
            '-b %s' % self.batchSize,
            '--lr %s' % self.learningRate,
            '--beta-control %s' % self.betaControl,
            '--beta cos',
            '--downfrac %s' % self.downFrac,
            '--valfrac %s' % self.valFrac,
            '--lamb %s' % self.lamb,
            '--bfactor %s' % self.bfactor,
            '--templateres %s' % self.templateres,
            '--split %s' % self._getExtraPath('sp-split.pkl'),
        ]

        if self.useMask:
            args.append('-r %s' % self.mask.get().getFileName())

        if self.relion31:
            args.append('--relion31')

        if len(self.getGpuList()) > 1:
            args.append('--multigpu')
            args.append('--num-gpus %s' % len(self.getGpuList()))

        if self.extraParams.hasValue():
            args.append(self.extraParams.get())

        return args
