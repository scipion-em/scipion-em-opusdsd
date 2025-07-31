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
from pwem.constants import ALIGN_PROJ, ALIGN_NONE
import pyworkflow.utils as pwutils
import pyworkflow.protocol.params as params
from pyworkflow.plugin import Domain
from pyworkflow.constants import PROD
from pwem.protocols import ProtProcessParticles, ProtFlexBase
from .. import Plugin
from ..constants import *

convertR = Domain.importFromPlugin('relion.convert', doRaise=True)

class OpusDsdProtTrain(ProtProcessParticles, ProtFlexBase):
    """
    Protocol to train OPUS-DSD neural network.
    """
    _label = 'opusdsd training'
    _devStatus = PROD

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _createFilenameTemplates(self):
        """ Centralize how files are called within the protocol. """
        myDict = {
            'input_parts': self._getExtra('input_particles.star'),
            'input_multiparts': self._getExtra('input_multiparticles.star'),
            'input_volume': self._getExtra('input_volume.mrc'),
            'input_mask': self._getExtra('input_mask.mrc'),
            'output_poses': self._getExtra('poses.pkl'),
            'input_multimask': self._getExtra('input_multimask_%(mask)d.mrc'),
            'output_ctfs': self._getExtra('ctfs.pkl'),
        }
        self._updateFilenamesDict(myDict)

    def _createFilenameTemplatesTraining(self):
        """ Centralize how files are called within the training protocol. """
        if not self.abInitio:
            myDict = {
                'workTrainDir': self._getOpusDSDTrainingProtocol()._getExtra()
            }
            self._updateFilenamesDict(myDict)

    # --------------------------- DEFINE param functions ----------------------

    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', params.PointerParam,
                      pointerClass="SetOfParticles, SetOfParticlesFlex",
                      label='Input Particles')

        form.addParam('abInitio', params.BooleanParam, default=True,
                      label="Ab-Initio condition",
                      help="If preprocess data is required, set to yes, if training data is required, set to no.")

        group = form.addGroup('Ab-Initio', condition='abInitio==%s' % True)
        group.addParam('useMask', params.BooleanParam, condition='abInitio==%s' % True, default=False,
                      label='Importing mask?',
                      help='If mask can be imported, set to yes, if not, set to no for mask creation from'
                           'a mandatory imported volume.')

        group.addParam('inputVolume', params.PointerParam, pointerClass='Volume', condition='useMask==%s' % False,
                      label='Input Volume',
                      help="The suggestion is to use a solvent mask created from this alrady provided volume. "
                           "Then, the program will focus on fitting the contents inside the created mask "
                           "(more specifically, the 2D projection of a 3D mask). Since the majority part of the "
                           "image doesn't contain electron density, using the original image size is wasteful. "
                           "By specifying a mask, our program will automatically determine a suitable crop rate "
                           "to keep only the region with densities.")

        group.addParam('threshold', params.FloatParam, default=0.008, condition='useMask==%s' % False,
                      label='Initial binarisation threshold',
                      expertLevel=params.LEVEL_ADVANCED,
                      help="This threshold is used to make an initial binary "
                           "mask from the average of the two unfiltered "
                           "half-reconstructions. If you don't know what "
                           "value to use, display one of the unfiltered "
                           "half-maps in a 3D surface rendering viewer and "
                           "find the lowest threshold that gives no noise "
                           "peaks outside the reconstruction.")

        group.addParam('extend', params.IntParam, default=2, condition='useMask==%s' % False,
                      label='Extend binary mask by (px)',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='The initial binary mask is extended this number of '
                           'pixels in all directions.')

        group.addParam('edge', params.IntParam, default=6, condition='useMask==%s' % False,
                      label='Add a soft-edge (px)',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='The extended binary mask is further extended with '
                           'a raised-cosine soft edge of the specified width.')

        group.addParam('inputMask', params.PointerParam, pointerClass='VolumeMask',
                      condition='useMask==%s' % True,
                      label="Input Mask",
                      help="The suggestion is to use an already given solvent mask. "
                           "If it isn't given, it must be calculated from the volume given, which will be necessary. "
                           "The program will focus on fitting the contents inside the mask (more specifically, "
                           "the 2D projection of a 3D mask). Since the majority part of the image doesn't contain "
                           "electron density, using the original image size is wasteful. By specifying a mask, "
                           "our program will automatically determine a suitable crop rate "
                           "to keep only the region with densities.")

        form.addParam('opusDSDTrainingProtocol', params.PointerParam,
                       condition='abInitio==%s' % False,
                       pointerClass='OpusDsdProtTrain',
                       label="Opus-DSD training protocol",
                       help="Previously executed 'opusdsd training'. "
                            "This will allow to load the necessary results the previous protocol achieved to get")

        form.addParam('Apix', params.FloatParam, default=-1,
                      label='Pixel size in A/pix',
                      help='If left as -1, pixel size will be the same as the sampling rate of the input particles.')

        form.addParam('boxSize', params.IntParam, default=-1,
                      label='Box size of reconstruction (pixels)',
                      help='If left as -1, box size will be the same as the dimensions of the input particles.')

        form.addSection(label='Training')
        group = form.addGroup('Multi-Body Training')
        group.addParam('multiBody', params.BooleanParam, default=False,
                       label="Multi-Body Training?",
                       help="If set to yes, a multi-body training will be performed, if set to no, then a single-body"
                            "training will be performed.")

        group.addParam('multiMasks', params.MultiPointerParam, pointerClass='VolumeMask',
                      condition='multiBody==%s' % True, allowsNull=True,
                      label='Multibody Masks',
                      help='Input of multiple masks of different bodies of the same input volume. Important for '
                           'later creation of a necessary star file for multi-body training.')

        group.addParam('zAffDim', params.IntParam, default=4,
                      condition='multiBody==%s ' % True,
                      validators=[params.Positive],
                      label='Dimension of latent variable for dynamics',
                      help='It is recommended to just set to default in case you are not sure what to type.')

        form.addParam('numEpochs', params.IntParam, default=20,
                      label='Number of epochs',
                      help='The number of epochs refers to the number '
                           'of full passes through the dataset for '
                           'training, and should be modified depending '
                           'on the number of particles in the dataset. '
                           'Even for non-ab-initio cases, the number of epochs should be left the same as previous '
                           'trainings.')

        form.addParam('zDim', params.IntParam, default=12,
                      validators=[params.Positive],
                      label='Dimension of latent variable',
                      help='It is recommended to first train on lower '
                           'resolution images (e.g. D=128) with '
                           '--zdim 1 and with --zdim 12 using the '
                           'default architecture (fast). Values between [1, 12].')

        group = form.addGroup('Encoder', expertLevel=params.LEVEL_ADVANCED)
        group.addParam('qLayers', params.IntParam, default=3,
                       label='Number of hidden layers of the encoder',
                       expertLevel=params.LEVEL_ADVANCED)
        group.addParam('qDim', params.IntParam, default=256,
                       label='Number of nodes in hidden layers of the encoder',
                       expertLevel=params.LEVEL_ADVANCED)

        group = form.addGroup('Decoder', expertLevel=params.LEVEL_ADVANCED)
        group.addParam('pLayers', params.IntParam, default=3,
                       label='Number of hidden layers of the decoder',
                       expertLevel=params.LEVEL_ADVANCED)
        group.addParam('pDim', params.IntParam, default=256,
                       label='Number of nodes in hidden layers of the decoder',
                       expertLevel=params.LEVEL_ADVANCED)

        form.addSection(label='Network parameters')
        form.addParam('batchSize', params.IntParam, default=8,
                      label='Batch size',
                      help='Batch size for processing images.')

        form.addParam('weightDecay', params.FloatParam, default=0.0,
                      label='Weight Decay',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Weight decay in Adam optimizer.')

        form.addParam('betaControl', params.FloatParam, default=1.,
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

        form.addParam('bfactor', params.FloatParam, default=3.75,
                      label='B-factor for reconstruction',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Reconstruction will be blurred by this factor, which corresponds to '
                           'exp(-bfactor/4 * s^2 * 4*pi^2) decaying to the FT of reconstruction. Possible '
                           'values between [3.,6.]. You may consider using higher values for more dynamic '
                           'structures.')

        form.addParam('learningRate', params.FloatParam, default=2e-4,
                      label='Learning rate',
                      help='Learning rate in Adam optimizer.')

        form.addParam('accumStep', params.IntParam, default=4,
                      label='Gradient accumulation', expertLevel=params.LEVEL_ADVANCED,
                      help='Gradient accumulation step for optimizer to increase the effective batch size. Best when '
                           'working with one gpu.')

        form.addParam('valFrac', params.FloatParam, default=0.2,
                      label='Validation image fraction',
                      help='Fraction of images held for validation.')

        form.addParam('downFrac', params.FloatParam, default=0.7,
                      label='Downsampling fraction',
                      help='Downsample to this fraction of original size. You can set it according to '
                           'resolution of consensus model and the templateres you set')

        form.addParam('templateres', params.IntParam, default=224,
                      label='Output size',
                      help='Define the output size of 3d volume of the convolutional network. You may keep it '
                           'around > D*downFrac, which is larger than the input size.')

        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       label="Choose GPU IDs",
                       help="GPU may have several cores. Set it to zero"
                            " if you do not know what we are talking about."
                            " First core index is 0, second 1 and so on."
                            " You can use multiple GPUs - in that case"
                            " set to i.e. *0 1 2*.")

        form.addParallelSection(threads=1, mpi=1)

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
        inParts = self._getInputParticles().getFileName()
        if self.multiBody:
            starFilename = self._getFileName('input_multiparts')
        else:
            starFilename = self._getFileName('input_parts')

        if inParts.endswith('.star'):
            shutil.copy(inParts, starFilename)
        else:
            alignType = ALIGN_PROJ if self._inputHasAlign() else ALIGN_NONE
            convertR.writeSetOfParticles(
                self._getInputParticles(), starFilename,
                outputDir=self._getExtra(),
                alignType=alignType)

        # Create links to binary files and write the .mrc file
        maskFilename = self._getFileName('input_mask')
        if self.useMask:
            inMask = self._getInputMask().getFileName()
            shutil.copy(inMask, maskFilename)
        else:
            volFilename = self._getFileName('input_volume')
            inVol = self._getInputVolume().getFileName()
            shutil.copy(inVol, volFilename)

            args = '--i %s ' % volFilename
            args += '--o %s ' % maskFilename
            args += '--ini_threshold %f ' % self.threshold
            args += '--extend_inimask %d ' % self.extend
            args += '--width_soft_edge %d ' % self.edge
            args += '--angpix %f' % self._getInputVolume().getSamplingRate()

            self._runProgram('relion_mask_create', args, fromRelion=True)

        # In case it's a multi rigid-body training, we create a starfile with all the mask parameters
        if self.multiBody:
            if not os.path.exists(self._getExtra('Masks')):
                pwutils.makePath(self._getExtra('Masks'))
            for i, mask in enumerate(self.multiMasks):
                multiMaskFilename = self._getFileName('input_multimask', mask=i)
                multiMask = mask.get().getFileName()
                shutil.copy(multiMask, multiMaskFilename)
            body_masks = [body for body in os.listdir(self._getExtra()) if body.startswith('input_multimask')]
            n_bodies = len(body_masks)
            for body in body_masks:
                shutil.move(self._getExtra(body), self._getExtra('Masks'))
            self._createMultiStarFile(n_bodies)

    def runParseMdStep(self):
        # Creating both poses and ctfs files for training and evaluation
        if self.multiBody:
            args = self._getFileName('input_multiparts')
        else:
            args = self._getFileName('input_parts')

        if self.boxSize.get() != -1:
            args += ' -D %d ' % self.boxSize.get()
        else:
            args += ' -D %d ' % self._getBoxSize()

        args += '--relion31 '

        if self.Apix.get() != -1:
            args += '--Apix %f ' % self.Apix.get()
        else:
            args += '--Apix %f ' % self._getInputParticles().getSamplingRate()

        args += '-o %s' % self._getFileName('output_poses')

        if not self.multiBody:
            self._runProgram('parse_pose_star', args)
        else:
            mask_params = [star for star in os.listdir(self._getExtra()) if star.endswith('bodies-tight-mask.star')]
            args += ' --masks %s ' % self._getExtra(mask_params[0])
            args += '--bodies %d' % int(mask_params[0][0])

            self._runProgram('parse_multi_pose_star', args)

        self._fixPosesTranslations()

        if self.multiBody:
            args = self._getFileName('input_multiparts')
        else:
            args = self._getFileName('input_parts')

        if self.Apix.get() != -1:
            args += ' --Apix %f ' % self.Apix.get()
        else:
            args += ' --Apix %f ' % self._getInputParticles().getSamplingRate()

        if self.boxSize.get() != -1:
            args += '-D %d ' % self.boxSize.get()
        else:
            args += '-D %d ' % self._getBoxSize()

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
        if self.multiBody:
            inputParticles = self._getFileName('input_multiparts')
        else:
            inputParticles = self._getFileName('input_parts')

        inputMask = self._getFileName('input_mask')

        if self.abInitio:
            initEpoch = self.numEpochs.get() - 2
        else:
            pwutils.cleanPath(self._getExtra())
            shutil.copytree(self._getFileName('workTrainDir'), self._getExtra())
            initEpoch = os.path.basename(self._getWorkDir()).split('.')[1]

        args = inputParticles
        args += ' --outdir %s ' % self._getExtra()
        args += '--ref_vol %s ' % inputMask
        args += '--zdim %d ' % self.zDim

        if self.multiBody:
            args += '--zaffdim %d ' % self.zDim

        outputPoses = self._getExtra('poses.pkl')
        args += '--poses %s ' % outputPoses

        if self.multiBody:
            args += '--zaffdim %d ' % self.zAffDim
            args += '--masks %s ' % self._getExtra('mask_params.pkl')

        outputCtfs = self._getExtra('ctfs.pkl')
        args += '--ctf %s ' % outputCtfs

        if not self.abInitio:
            weights = self._getExtra(f'Results.{initEpoch}/weights.{initEpoch}.pkl')
            z = self._getExtra(f'Results.{initEpoch}/z.{initEpoch}.pkl')
            args += '--load %s ' % weights
            args += '--latents %s ' % z

        args += '--split %s ' % self._getExtra('sp-split.pkl')
        args += '--valfrac %f ' % self.valFrac
        args += '--verbose '
        args += '--relion31 '
        args += '--lazy-single '

        if self.abInitio:
            args += '--num-epochs %d ' % self.numEpochs
        else:
            totalEpochs = self._getEpoch(initEpoch) + 2
            args += '--num-epochs %d ' % totalEpochs

        args += '--batch-size %d ' % self.batchSize

        if self.weightDecay.get() != 0:
            args += '--wd %f ' % self.weightDecay

        args += '--lr %f ' % self.learningRate
        args += '--accum-step %d ' % self.accumStep
        args += '--lamb %f ' % self.lamb

        if self.multiBody:
            if self.downFrac.get() * (self._getBoxSize() - 1) >= 128:
                args += '--downfrac %f ' % self.downFrac
            else:
                raise ValueError("Error while asserting, please change the downsampling factor accordingly, as "
                                 "the product between the factor and the original size of the particles are not above 128")
        else:
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

        self._outputRegroup(initEpoch)

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

    def _getInputMask(self):
        return self.inputMask.get()

    def _getBoxSize(self):
        return self._getInputParticles().getXDim()

    def _getExtra(self, *paths):
        return os.path.abspath(self._getExtraPath(*paths))

    def _getWorkDir(self):
        workDir = [dir for dir in os.listdir(self._getExtra()) if dir.startswith('Results')][0]
        return self._getExtra(workDir)

    def _runProgram(self, program, args, fromRelion=False):
        gpus = ','.join(str(i) for i in self.getGpuList())
        if not fromRelion:
            self.runJob(Plugin.getProgram(program, gpus, fromCryodrgn=True), args)
        else:
            self.runJob(Plugin.getRelionProgram(program), args)

    def _inputHasAlign(self):
        return self._getInputParticles().hasAlignmentProj()

    def _createMultiStarFile(self, n_bodies):
        with open(self._getExtra(f'{n_bodies}-bodies-tight-mask.star'), 'w') as f:
            f.write("data_\n\n")
            f.write("loop_\n")
            f.write("_rlnBodyMaskName\n"
                    "_rlnBodyRotateRelativeTo\n"
                    "_rlnBodySigmaAngles\n"
                    "_rlnBodySigmaOffset\n"
                    "_rlnBodyReferenceName\n")
            for mask in masks_info:
                f.write(f"{mask['mask_name']} "
                        f"{mask['rotate_relative_to']} "
                        f"{mask['sigma_angles']} "
                        f"{mask['sigma_offset']} "
                        f"{mask['reference_name']}\n")

    def _fixPosesTranslations(self):
        # In case poses.pkl file has 3 dimensions, we set them to 2 so that the program is able to read it.
        posesFile = self._getFileName('output_poses')
        with open(posesFile, 'rb') as f:
            if self.multiBody:
                rot, trans, euler, *_ = pickle.load(f)
            else:
                rot, trans, euler = pickle.load(f)
        N, D = trans.shape
        if D > 2:
            print(f'Truncating dimensions from {D}D to 2D.')
            trans = trans[:, :2]
        with open(posesFile, 'wb') as f:
            pickle.dump((rot, trans, euler), f)

    def _getEpoch(self, initEpoch):
        """ Return the specific analysis iteration. """
        if self.abInitio:
            self._epoch = int(initEpoch)
        else:
            self._epoch = self.numEpochs.get() + int(initEpoch)
        return self._epoch

    def _outputRegroup(self, initEpoch):
        # Eliminating previous Results directory to focus on next training (just in not-ab-initio case)
        if not self.abInitio:
            workDir = [dir for dir in os.listdir(self._getExtra()) if dir.startswith('Results')][0]
            pwutils.cleanPath(self._getExtra(workDir))
        # Creating outputs for the evaluated results from training
        outputFolder = self._getExtra(f'Results.{self._getEpoch(initEpoch)}')
        os.makedirs(outputFolder, exist_ok=True)
        files = [file for file in os.listdir(self._getExtra()) if len(file.split('.')) == 3]
        for file in files:
            if file.endswith('.pkl') and int(file.split('.')[1]) == self._getEpoch(initEpoch):
                shutil.move(self._getExtra(file), os.path.join(outputFolder, file))
            elif file.endswith('.pkl') and int(file.split('.')[1]) != self._getEpoch(initEpoch):
                os.remove(self._getExtra(file))

    def _getOpusDSDTrainingProtocol(self):
        return self.opusDSDTrainingProtocol.get()