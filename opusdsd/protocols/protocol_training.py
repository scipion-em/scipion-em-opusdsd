# ***********************************************************************
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
from pyworkflow.protocol.constants import *
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
                      allowsNull=True, label='Input Volume',
                      help="The suggestion is to use a solvent mask created from this alrady provided volume. "
                           "Then, the program will focus on fitting the contents inside the created mask "
                           "(more specifically, the 2D projection of a 3D mask). Since the majority part of the "
                           "image doesn't contain electron density, using the original image size is wasteful. "
                           "By specifying a mask, our program will automatically determine a suitable crop rate "
                           "to keep only the region with densities.")

        group.addParam('threshold', params.FloatParam, default=0.01, condition='useMask==%s' % False,
                      label='Initial binarisation threshold',
                      expertLevel=params.LEVEL_ADVANCED,
                      help="This threshold is used to make an initial binary "
                           "mask from the average of the two unfiltered "
                           "half-reconstructions. If you don't know what "
                           "value to use, display one of the unfiltered "
                           "half-maps in a 3D surface rendering viewer and "
                           "find the lowest threshold that gives no noise "
                           "peaks outside the reconstruction.")

        group.addParam('inputMask', params.PointerParam, pointerClass='VolumeMask',
                      condition='useMask==%s' % True, allowsNull=True,
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

        form.addSection(label='Training')
        group = form.addGroup('Multi-Body Training', condition='abInitio==%s' % True)
        group.addParam('multiBody', params.BooleanParam, default=False,
                       condition='abInitio==%s' % True,
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
                      condition='abInitio==%s' % True,
                      label='Dimension of latent variable',
                      help='It is recommended to first train on lower '
                           'resolution images (e.g. D=128) with '
                           '--zdim 1 and with --zdim 12 using the '
                           'default architecture (fast). Values between [1, 12].')

        form.addParam('lazyLoad', params.BooleanParam, default=False,
                      label='Lazy loading particles into memory',
                      help='If set True, all particles will be loaded into memory, which would consume more memory but less '
                           'CPU capacity. If set False, it will take into account by default 8 workers in CPU, so it will '
                           'consume less memory but slightly more CPU capacity.')

        group = form.addGroup('Encoder', condition='abInitio==%s' % True, expertLevel=params.LEVEL_ADVANCED)
        group.addParam('qLayers', params.IntParam, default=3,
                       condition='abInitio==%s' % True,
                       label='Number of hidden layers of the encoder',
                       expertLevel=params.LEVEL_ADVANCED)
        group.addParam('qDim', params.IntParam, default=256,
                       condition='abInitio==%s' % True,
                       label='Number of nodes in hidden layers of the encoder',
                       expertLevel=params.LEVEL_ADVANCED)

        group = form.addGroup('Decoder', condition='abInitio==%s' % True, expertLevel=params.LEVEL_ADVANCED)
        group.addParam('pLayers', params.IntParam, default=3,
                       condition='abInitio==%s' % True,
                       label='Number of hidden layers of the decoder',
                       expertLevel=params.LEVEL_ADVANCED)
        group.addParam('pDim', params.IntParam, default=256,
                       condition='abInitio==%s' % True,
                       label='Number of nodes in hidden layers of the decoder',
                       expertLevel=params.LEVEL_ADVANCED)

        form.addSection(label='Network parameters')
        form.addParam('batchSize', params.IntParam, default=8,
                      condition='abInitio==%s' % True,
                      label='Batch size',
                      help='Batch size for processing images.')

        form.addParam('weightDecay', params.FloatParam, default=0.0,
                      condition='abInitio==%s' % True,
                      label='Weight Decay',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Weight decay in Adam optimizer.')

        form.addParam('betaControl', params.FloatParam, default=1.,
                      condition='abInitio==%s' % True,
                      label='Beta restraint strength for KL target',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Beta parameter that controls the strength of the beta-VAE prior. The larger '
                           'the argument, the stronger the strength of the standard Gaussian restraint.')

        form.addParam('lamb', params.FloatParam, default=0.5,
                      condition='abInitio==%s' % True,
                      label='Restraint strength for umap prior',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='This controls the stretch of the UMAP-inspired prior for '
                           'the encoder network that encourages the encoding of structural '
                           'information for images in the same projection class. Possible values between [0.1, 3.].')

        form.addParam('bfactor', params.FloatParam, default=3.75,
                      condition='abInitio==%s' % True,
                      label='B-factor for reconstruction',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Reconstruction will be blurred by this factor, which corresponds to '
                           'exp(-bfactor/4 * s^2 * 4*pi^2) decaying to the FT of reconstruction. Possible '
                           'values between [3.,6.]. You may consider using higher values for more dynamic '
                           'structures.')

        form.addParam('learningRate', params.FloatParam, default=1e-4,
                      condition='abInitio==%s' % True,
                      label='Learning rate',
                      help='Learning rate in Adam optimizer.')

        form.addParam('accumStep', params.IntParam, default=4,
                      condition='abInitio==%s' % True,
                      label='Gradient accumulation', expertLevel=params.LEVEL_ADVANCED,
                      help='Gradient accumulation step for optimizer to increase the effective batch size. Best when '
                           'working with one gpu.')

        form.addParam('valFrac', params.FloatParam, default=0.2,
                      condition='abInitio==%s' % True,
                      label='Validation image fraction',
                      help='Fraction of images held for validation.')

        form.addParam('downFrac', params.FloatParam, default=1.0,
                      condition='abInitio==%s' % True,
                      label='Downsampling fraction', expertLevel=params.LEVEL_ADVANCED,
                      help='Downsample to this fraction of original size. You can set it according to '
                           'resolution of consensus model and the templateres you set')

        form.addParam('templateres', params.IntParam, default=144,
                      condition='abInitio==%s' % True,
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
        run = self if self.abInitio else self._getOpusDSDTrainingProtocol()

        inParts = self._getInputParticles().getFileName()
        if run.multiBody:
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
        if self.abInitio:
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
                args += '--substitute binarize %f ' % self.threshold

                self._runProgram('xmipp_transform_threshold', args, fromXmipp=True)

        # In case it's a multi rigid-body training, we create a starfile with all the mask parameters
        if run.multiBody:
            if not os.path.exists(self._getExtra('Masks')):
                pwutils.makePath(self._getExtra('Masks'))
            for i, mask in enumerate(run.multiMasks):
                multiMaskFilename = self._getFileName('input_multimask', mask=i)
                multiMask = mask.get().getFileName()
                shutil.copy(multiMask, multiMaskFilename)
            body_masks = [body for body in os.listdir(self._getExtra()) if body.startswith('input_multimask')]
            n_bodies = len(body_masks)
            for body in body_masks:
                shutil.move(self._getExtra(body), self._getExtra('Masks'))
            self._createMultiStarFile(n_bodies)

    def runParseMdStep(self):
        # Creating both poses and ctf files for training and evaluation
        run = self if self.abInitio else self._getOpusDSDTrainingProtocol()
        args = self._getParsePoseCtfArgs()

        if not run.multiBody:
            self._runProgram('parse_pose_star', args[0])
        else:
            self._runProgram('parse_multi_pose_star', args[0])

        self._fixPosesTranslations()

        self._runProgram('parse_ctf_star', args[1])

    def runTrainingStep(self):
        # Training step for Opus-DSD
        run = self if self.abInitio else self._getOpusDSDTrainingProtocol()
        args = self._getTrainingArgs()

        if self.abInitio:
            if not run.multiBody:
                self._runProgram('train_cv', args)
            else:
                self._runProgram('train_multi', args)

            self._outputRegroup(self.numEpochs.get() - 2)
        else:
            if not run.multiBody:
                self._runProgram('train_cv', args[0])
            else:
                self._runProgram('train_multi', args[0])

            self._outputRegroup(args[1])

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

    # --------------------------- ARGS functions -----------------------------

    def _getParsePoseCtfArgs(self):
        run = self if self.abInitio else self._getOpusDSDTrainingProtocol()

        if run.multiBody:
            pose_args = self._getFileName('input_multiparts')
        else:
            pose_args = self._getFileName('input_parts')

        pose_args += ' -D %d ' % self._getBoxSize()
        pose_args += '--relion31 '
        pose_args += '--Apix %f ' % self._getInputParticles().getSamplingRate()
        pose_args += '-o %s' % self._getFileName('output_poses')

        if run.multiBody:
            mask_params = [star for star in os.listdir(self._getExtra()) if star.endswith('bodies-tight-mask.star')]
            pose_args += ' --masks %s ' % self._getExtra(mask_params[0])
            pose_args += '--bodies %d' % int(mask_params[0][0])

        if run.multiBody:
            ctf_args = self._getFileName('input_multiparts')
        else:
            ctf_args = self._getFileName('input_parts')

        ctf_args += ' --Apix %f ' % self._getInputParticles().getSamplingRate()
        ctf_args += '-D %d ' % self._getBoxSize()
        ctf_args += '--relion31 '
        ctf_args += '-o %s ' % self._getFileName('output_ctfs')

        acquisition = self._getInputParticles().getAcquisition()

        ctf_args += '--kv %f ' % acquisition.getVoltage()
        ctf_args += '--cs %f ' % acquisition.getSphericalAberration()
        ctf_args += '-w %f ' % acquisition.getAmplitudeContrast()
        ctf_args += '--ps 0.'  # required due to OPUS-DSD parsing bug

        return pose_args, ctf_args

    def _getTrainingArgs(self):
        run = self if self.abInitio else self._getOpusDSDTrainingProtocol()

        if run.multiBody:
            inputParticles = self._getFileName('input_multiparts')
        else:
            inputParticles = self._getFileName('input_parts')

        inputMask = self._getFileName('input_mask')

        args = inputParticles

        if self.abInitio:
            files = [file for file in os.listdir(self._getExtra()) if file.startswith('weights')]
            if len(files) > 1:
                initEpoch = max([int(os.path.basename(self._getExtra(file)).split('.')[1]) for file in files])
                weights = self._getExtra(f'weights.{initEpoch}.pkl')
                z = self._getExtra(f'z.{initEpoch}.pkl')
                args += ' --load %s ' % weights
                args += '--latents %s' % z

            args += ' --num-epochs %d ' % self.numEpochs
        else:
            files = [file for file in os.listdir(self._getExtra()) if file.startswith('weights')]
            if len(files) > 1:
                prevEpoch = os.path.basename(self._getWorkDir()).split('.')[1]
                initEpoch = max([int(os.path.basename(os.path.abspath(file)).split('.')[1]) for file in files])

                totalEpochs = self._getEpoch(prevEpoch) + 2
                args += ' --num-epochs %d ' % totalEpochs
            else:
                pwutils.cleanPath(self._getExtra())
                shutil.copytree(self._getFileName('workTrainDir'), self._getExtra())
                initEpoch = os.path.basename(self._getWorkDir()).split('.')[1]

                totalEpochs = self._getEpoch(initEpoch) + 2
                args += ' --num-epochs %d ' % totalEpochs

            weights = self._getExtra(f'Results.{initEpoch}/weights.{initEpoch}.pkl')
            z = self._getExtra(f'Results.{initEpoch}/z.{initEpoch}.pkl')
            args += '--load %s ' % weights
            args += '--latents %s ' % z

        args += '--outdir %s ' % self._getExtra()
        args += '--ref_vol %s ' % inputMask
        args += '--zdim %d ' % run.zDim

        outputPoses = self._getExtra('poses.pkl')
        args += '--poses %s ' % outputPoses

        if run.multiBody:
            args += '--zaffdim %d ' % run.zAffDim
            args += '--masks %s ' % self._getExtra('mask_params.pkl')

        outputCtfs = self._getExtra('ctfs.pkl')
        args += '--ctf %s ' % outputCtfs

        args += '--split %s ' % self._getExtra('sp-split.pkl')
        args += '--valfrac %f ' % run.valFrac
        args += '--verbose '
        args += '--relion31 '
        args += '--lazy-single '

        if self.lazyLoad:
            args += '--inmem '

        args += '--batch-size %d ' % run.batchSize

        if run.weightDecay.get() != 0:
            args += '--wd %f ' % run.weightDecay

        args += '--lr %f ' % run.learningRate
        args += '--accum-step %d ' % run.accumStep
        args += '--lamb %f ' % run.lamb

        if run.multiBody:
            if run.downFrac.get() * (self._getBoxSize() - 1) >= 128:
                args += '--downfrac %f ' % run.downFrac
            else:
                raise ValueError("Error while asserting, please change the downsampling factor accordingly, as "
                                 "the product between the factor and the original size of the particles are not above 128")
        else:
            args += '--downfrac %f ' % run.downFrac

        args += '--templateres %d ' % run.templateres
        args += '--bfactor %f ' % run.bfactor
        args += '--beta cos '
        args += '--beta-control %f ' % run.betaControl
        args += '--num-gpus %d ' % len(self.getGpuList())
        args += '--enc-layers %d ' % run.qLayers
        args += '--enc-dim %d ' % run.qDim
        args += '--encode-mode grad '
        args += '--dec-layers %d ' % run.pLayers
        args += '--dec-dim %d ' % run.pDim
        args += '--pe-type vanilla '
        args += '--template-type conv '
        args += '--activation relu'

        if self.abInitio:
            return args
        else:
            if len(files) > 1:
                return args, prevEpoch
            else:
                return args, initEpoch

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

    def _runProgram(self, program, args, fromXmipp=False):
        gpus = ','.join(str(i) for i in self.getGpuList())
        if not fromXmipp:
            self.runJob(Plugin.getProgram(program, gpus, fromCryodrgn=True), args, env=pwutils.Environ())
        else:
            self.runJob(Plugin.getXmippProgram(program), args)

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
        run = self if self.abInitio else self._getOpusDSDTrainingProtocol()
        posesFile = self._getFileName('output_poses')
        with open(posesFile, 'rb') as f:
            if run.multiBody:
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

    def _outputRegroup(self, epoch):
        # Eliminating previous Results directory to focus on next training (just in not-ab-initio case)
        if not self.abInitio:
            workDir = [dir for dir in os.listdir(self._getExtra()) if dir.startswith('Results')][0]
            pwutils.cleanPath(self._getExtra(workDir))
        # Creating outputs for the evaluated results from training
        outputFolder = self._getExtra(f'Results.{self._getEpoch(epoch)}')
        os.makedirs(outputFolder, exist_ok=True)
        files = [file for file in os.listdir(self._getExtra()) if len(file.split('.')) == 3]
        for file in files:
            if file.endswith('.pkl') and int(file.split('.')[1]) == self._getEpoch(epoch):
                shutil.move(self._getExtra(file), os.path.join(outputFolder, file))
            elif file.endswith('.pkl') and int(file.split('.')[1]) != self._getEpoch(epoch):
                os.remove(self._getExtra(file))

    def _getOpusDSDTrainingProtocol(self):
        return self.opusDSDTrainingProtocol.get()