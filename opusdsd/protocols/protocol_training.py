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
import os, shutil
import pickle
import numpy as np
import re
from glob import glob

from pwem.constants import ALIGN_PROJ, ALIGN_NONE
import pyworkflow.utils as pwutils
import pyworkflow.protocol.params as params
import pyworkflow.object as pwobj
import pwem.objects.data_flexhub as pwobjflex
from pyworkflow.plugin import Domain
from pyworkflow.constants import PROD

from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.image_handler import ImageHandler
from xmipp3.convert import readSetOfParticles, writeSetOfParticles

from pwem.protocols import ProtProcessParticles, ProtFlexBase
import pwem.objects as emobj
from .protocol_analyze import OpusDsdProtAnalyze

from .. import Plugin
from ..constants import *

KMEANS = 0
PCS = 1
PREPROCESS = True
ANALYSIS = False

convertR = Domain.importFromPlugin('relion.convert', doRaise=True)

class OpusDsdProtTrain(ProtProcessParticles, ProtFlexBase):
    """
    Protocol to train OPUS-DSD neural network.
    """
    _label = 'training'
    _devStatus = PROD

    def _createFilenameTemplates(self):
        """ Centralize how files are called within the protocol. """
        myDict = {
            'input_parts': self._getExtra('output_images.star'),
            'input_volume': self._getExtra('input_volume.mrc'),
            'output_poses': self._getExtra('poses.pkl'),
            'output_ctfs': self._getExtra('ctfs.pkl'),
        }
        self._updateFilenamesDict(myDict)

    def _createFilenameTemplatesTraining(self):
        """ Centralize how files are called within the analysis protocol. """
        if not self.abInitio:
            myDict = {
                'workAnalysisDir': self._getOpusDSDAnalysisProtocol()._getExtra()
            }
            self._updateFilenamesDict(myDict)

    # --------------------------- DEFINE param functions ----------------------

    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       label="Choose GPU IDs",
                       help="GPU may have several cores. Set it to zero"
                            " if you do not know what we are talking about."
                            " First core index is 0, second 1 and so on."
                            " You can use multiple GPUs - in that case"
                            " set to i.e. *0 1 2*.")

        form.addParam('inputParticles', params.PointerParam,
                      pointerClass="SetOfParticles, SetOfParticlesFlex",
                      label='OPUS-DSD particles')

        group = form.addGroup('Volume selection')
        group.addParam('useVolume', params.BooleanParam, default=True,
                       label="Use volumes?")

        group.addParam('inputVolume', params.PointerParam, pointerClass='Volume', condition='useVolume',
                       label="OPUS-DSD Volume",
                       help="The suggestion is to use a solvent volume created from consensus model. The program will "
                            "focus on fitting the contents inside the volume (more specifically, "
                            "the 2D projection of a 3D mask). Since the majority part of image "
                            "doesn't contain electron density, using the original image size is "
                            "wasteful, by specifying a mask, our program will automatically "
                            "determine a suitable crop rate to keep only the region with densities.")

        form.addSection(label='Preprocess')
        form.addParam('Apix', params.FloatParam, default=-1,
                      label='Pixel size in A for output volumes',
                      help='If left as -1, pixel size will be the same as the sampling rate of the input particles.')

        form.addParam('boxSize', params.IntParam, default=-1,
                      label='Box size of reconstruction (pixels)',
                      help='If left as -1, box size will be the same as the dimensions of the input particles.')

        form.addParam('relion31', params.BooleanParam, default=True,
                      label="Are particles from RELION 3.1?")

        form.addParallelSection(threads=16, mpi=0)
        form.getParam('numberOfThreads').default = pwobj.Integer(1)

        form.addSection(label='Training')

        group = form.addGroup('Protocols')
        group.addParam('abInitio', params.BooleanParam, default=True,
                       label="Ab-Initio condition",
                       help="If preprocess data is required, set to yes, if analysis data is required, set to no.")

        group.addParam('opusDSDAnalysisProtocol', params.PointerParam,
                       condition='abInitio==%s' % ANALYSIS,
                       pointerClass='OpusDsdProtAnalyze',
                       label="Opus-DSD analysis protocol",
                       help="Previously executed 'Analyze - Opus-DSD'. "
                            "This will allow to load the necessary results the previous protocol achieved to get")

        form.addParam('multiBody', params.BooleanParam, default=False,
                       label="Multi-Body Training",
                       expertLevel=params.LEVEL_ADVANCED,
                       help="If set to yes, a multi-body training will be performed, if set to no, then a single-body"
                            "training will be performed.")

        form.addParam('numEpochs', params.IntParam, default=20,
                      label='Number of epochs',
                      help='The number of epochs refers to the number '
                           'of full passes through the dataset for '
                           'training, and should be modified depending '
                           'on the number of particles in the dataset. '
                           'For a 100k particle dataset, the above '
                           'settings required ~6 min per epoch for D=128 '
                           'images + default architecture, ~12 min/epoch '
                           'for D=128 images + large architecture, and ~47 '
                           'min per epoch for D=256 images + large architecture.')

        form.addParam('zDim', params.IntParam, default=10,
                      validators=[params.Positive],
                      label='Dimension of latent variable',
                      help='It is recommended to first train on lower '
                           'resolution images (e.g. D=128) with '
                           '--zdim 1 and with --zdim 10 using the '
                           'default architecture (fast). Values between [1, 10].')

        group = form.addGroup('Encoder', expertLevel=params.LEVEL_ADVANCED)
        group.addParam('qLayers', params.IntParam, default=3,
                       label='Number of hidden layers of the encoder',
                       expertLevel=params.LEVEL_ADVANCED)
        group.addParam('qDim', params.IntParam, default=1024,
                       label='Number of nodes in hidden layers of the encoder',
                       expertLevel=params.LEVEL_ADVANCED)

        group = form.addGroup('Decoder', expertLevel=params.LEVEL_ADVANCED)
        group.addParam('pLayers', params.IntParam, default=3,
                       label='Number of hidden layers of the decoder',
                       expertLevel=params.LEVEL_ADVANCED)
        group.addParam('pDim', params.IntParam, default=1024,
                       label='Number of nodes in hidden layers of the decoder',
                       expertLevel=params.LEVEL_ADVANCED)

        form.addSection(label='Advanced')
        form.addParam('batchSize', params.IntParam, default=4,
                      label='Batch size',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Batch size for processing images.')

        form.addParam('betaControl', params.FloatParam, default=0.5,
                      label='Beta restraint strength for KL target',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Beta parameter that controls the strength of the beta-VAE prior. The larger '
                           'the argument, the stronger the strength of the standard Gaussian restraint.')

        form.addParam('lamb', params.FloatParam, default=0.5,
                      label='Restraint strength for umap prior',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='This controls the stretch of the UMAP-inspired prior for '
                           'the encoder network that encourages the encoding of structural '
                           'information for images in the same projection class. Possible values between [0.1, 3.].')

        form.addParam('bfactor', params.FloatParam, default=4.,
                      label='B-factor for reconstruction',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Reconstruction will be blurred by this factor, which corresponds to '
                           'exp(-bfactor/4 * s^2 * 4*pi^2) decaying to the FT of reconstruction. Possible '
                           'values between [3.,6.]. You may consider using higher values for more dynamic '
                           'structures.')

        form.addParam('learningRate', params.FloatParam, default=1e-8,
                      label='Learning rate', expertLevel=params.LEVEL_ADVANCED,
                      help='Learning rate in Adam optimizer.')

        form.addParam('valFrac', params.FloatParam, default=0.1,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Validation image fraction',
                      help='Fraction of images held for validation.')

        form.addParam('downFrac', params.FloatParam, default=0.5,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Downsampling fraction',
                      help='Downsample to this fraction of original size.')

        form.addParam('templateres', params.IntParam, default=192,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Output size',
                      help='Define the output size of 3d volume of the convolutional network. You may keep it '
                           'around D*downFrac/0.75, which is larger than the input size.')

    # --------------------------- INSERT steps functions ----------------------

    def _insertAllSteps(self):
        self._createFilenameTemplates()
        self._createFilenameTemplatesTraining()

        if self.abInitio:
            self._insertFunctionStep(self.convertInputStep)
            self._insertFunctionStep(self.runParseMdStep)

        self._insertFunctionStep(self.runTrainingStep)

    # --------------------------- STEPS functions -----------------------------

    def convertInputStep(self):
        """ Create a star file as expected by OPUS-DSD."""
        alignType = ALIGN_PROJ if self._inputHasAlign() else ALIGN_NONE
        inSet = self._getInputParticles()
        starFilename = self._getFileName('input_parts')
        convertR.writeSetOfParticles(
            inSet, starFilename,
            outputDir=self._getExtra(),
            alignType=alignType)

        # Create links to binary files and write the .mrc file
        if self.useVolume:
            ih = ImageHandler()
            inVol = self._getInputVolume().getFileName()
            ih.convert(inVol, self._getFileName('input_volume'))

    def runParseMdStep(self):
        # Creating both poses and ctfs files for training and evaluation
        if self.boxSize.get() != -1:
            args = '-D %d ' % self.boxSize.get()
        else:
            args = '-D %d ' % self._getBoxSize()

        if self.relion31:
            args += '--relion31 '

        if self.Apix.get() != -1:
            args += '--Apix %f ' % self.Apix.get()
        else:
            args += '--Apix %f ' % self._getInputParticles().getSamplingRate()

        args += '-o %s ' % self._getFileName('output_poses')
        args += '--outdir %s ' % self._getExtra()
        args += self._getFileName('input_parts')

        self._runProgram('parse_pose_star', args)

        self._fixPosesTranslations()

        args = self._getFileName('input_parts')

        if self.Apix.get() != -1:
            args += ' --Apix %f ' % self.Apix.get()
        else:
            args += ' --Apix %f ' % self._getInputParticles().getSamplingRate()

        if self.boxSize.get() != -1:
            args += '-D %d ' % self.boxSize.get()
        else:
            args += '-D %d ' % self._getBoxSize()

        if self.relion31:
            args += '--relion31 '

        args += '-o %s ' % self._getFileName('output_ctfs')

        acquisition = self._getInputParticles().getAcquisition()

        args += '--kv %f ' % acquisition.getVoltage()
        args += '--cs %f ' % acquisition.getSphericalAberration()
        args += '-w %f ' % acquisition.getAmplitudeContrast()
        args += '--ps 0.'  # required due to OPUS-DSD parsing bug

        self._runProgram('parse_ctf_star', args)

    def runTrainingStep(self):
        # Call OPUS-DSD with the appropriate parameters
        epoch = self.numEpochs.get() - 2

        if self.abInitio:
            inputParticles = self._getExtra('input/*.mrcs')
            outputPoses = self._getExtra('poses.pkl')
        else:
            pwutils.cleanPath(self._getExtra())
            shutil.copytree(self._getFileName('workAnalysisDir'), self._getExtra())
            inputParticles = self._getExtra('input/*.mrcs')
            outputPoses = self._getExtra('CVResults.%d/pose.%d.pkl') % (epoch, epoch)

        args = inputParticles
        args += ' --outdir %s ' % self._getExtra()

        if self.useVolume:
            inputVolume = self._getFileName('input_volume')
            args += '--ref_vol %s ' % inputVolume

        args += '--zdim %d ' % self.zDim

        if self.multiBody:
            args += '--zaffdim %d ' % self.zDim

        args += '--poses %s ' % outputPoses

        outputCtfs = self._getExtra('ctfs.pkl')
        args += '--ctf %s ' % outputCtfs

        if not self.abInitio:
            weights = self._getExtra('CVResults.%d/weights.%d.pkl') % (epoch, epoch)
            args += '--load %s ' % weights

        args += '--split %s ' % self._getExtra('sp-split.pkl')
        args += '--valfrac %f ' % self.valFrac

        if self.relion31:
            args += '--relion31 '

        args += '--lazy-single '
        args += '--num-epochs %d ' % self.numEpochs
        args += '--batch-size %d ' % self.batchSize
        args += '--lr %f ' % self.learningRate
        args += '--lamb %f ' % self.lamb
        args += '--downfrac %f ' % self.downFrac
        args += '--templateres %d ' % self.templateres
        args += '--bfactor %f ' % self.bfactor
        args += '--beta cos '
        args += '--beta-control %f ' % self.betaControl
        args += '--num-gpus %d ' % len(self.getGpuList())
        args += '--enc-layers %d ' % self.qLayers
        args += '--enc-dim %d ' % self.qDim
        args += '--encode-mode grad '
        args += '--dec-layers %d ' % self.pLayers
        args += '--dec-dim %d ' % self.pDim
        args += '--pe-type vanilla '
        args += '--template-type conv '
        args += '--activation relu'

        if not self.multiBody:
            self._runProgram('train_cv', args)
        else:
            self._runProgram('train_multi', args)

        self._outputRegroup()

    # --------------------------- INFO functions ------------------------------

    def _summary(self):
        summary = ["Training CV for %d epochs." % self.numEpochs]

        return summary

    def _validateBase(self):
        errors = []

        if self._getBoxSize() % 2 != 0:
            errors.append("Box size must be even!")

        if not self._inputHasAlign():
            errors.append("Input particles have no alignment!")

        if self._getBoxSize() < 128:
            errors.append("OPUS-DSD requires a box size > 128 x 128 pixels.")

        return errors

    # --------------------------- UTILS functions -----------------------------

    def _getInputParticles(self):
        return self.inputParticles.get()

    def _getInputVolume(self):
        return self.inputVolume.get()

    def _getBoxSize(self):
        return self._getInputParticles().getXDim()

    def _getExtra(self, *paths):
        return os.path.abspath(self._getExtraPath(*paths))

    def _runProgram(self, program, args, fromCryodrgn=True):
        gpus = ','.join(str(i) for i in self.getGpuList())
        self.runJob(Plugin.getProgram(program, gpus, fromCryodrgn=fromCryodrgn), args)

    def _inputHasAlign(self):
        return self._getInputParticles().hasAlignmentProj()

    def _fixPosesTranslations(self):
        # In case poses.pkl file has 3 dimensions, we set them to 2 so that the program is able to read it.
        posesFile = self._getFileName('output_poses')
        with open(posesFile, 'rb') as f:
            rot, trans, euler = pickle.load(f)
        N, D = trans.shape
        if D > 2:
            print(f'Truncating dimensions from {D}D to 2D.')
            trans = trans[:, :2]
        with open(posesFile, 'wb') as f:
            pickle.dump((rot, trans, euler), f)

    def _outputRegroup(self):
        # Creating outputs for the evaluated results from training
        output = self._getExtra()
        for i in range(self.numEpochs.get()):
            outputFolder = self._getExtra(f'CVResults.{i}')
            os.makedirs(outputFolder, exist_ok=True)
            outpose = f'{output}/pose.{i}.pkl'
            if os.path.exists(outpose):
                shutil.move(outpose, os.path.join(outputFolder, f'pose.{i}.pkl'))
            outz = f'{output}/z.{i}.pkl'
            if os.path.exists(outz):
                shutil.move(outz, os.path.join(outputFolder, f'z.{i}.pkl'))
            outweights = f'{output}/weights.{i}.pkl'
            if os.path.exists(outweights):
                shutil.move(outweights, os.path.join(outputFolder, f'weights.{i}.pkl'))

    def _getOpusDSDAnalysisProtocol(self):
        return self.opusDSDAnalysisProtocol.get()