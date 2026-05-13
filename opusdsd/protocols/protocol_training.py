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

    class OpusDsdProtTrain(ProtProcessParticles, ProtFlexBase):
        """
        Trains the OPUS-DSD neural network for cryo-EM continuous heterogeneity analysis.
        The protocol is designed to preprocess particle datasets, prepare pose and CTF
        metadata, and execute neural network training workflows for both single-body and
        multi-body structural dynamics.

        AI Generated:

        OPUS-DSD Training (OpusDsdProtTrain) — User Manual
            Overview

            The OPUS-DSD Training protocol provides a deep learning framework for modeling
            structural variability in cryo-EM datasets. Its primary goal is to train a neural
            network capable of learning latent representations of molecular flexibility from
            aligned particle images. The protocol integrates preprocessing, metadata parsing,
            latent-space configuration, and neural network optimization into a unified workflow
            suitable for continuous heterogeneity analysis.

            In biological applications, the protocol is commonly used to study conformational
            landscapes, domain motions, and flexible assemblies that cannot be accurately
            represented by a single static reconstruction. By learning a low-dimensional latent
            representation, OPUS-DSD enables exploration of structural transitions and dynamic
            variability directly from particle images.

            Inputs and General Workflow

            The protocol requires a set of aligned cryo-EM particles as the main input.
            Particle metadata are converted into STAR files compatible with OPUS-DSD processing.
            During preprocessing, pose and CTF information are extracted and reformatted so the
            neural network can correctly interpret image orientations and microscope parameters.

            The workflow supports both ab-initio training and continuation of previous training
            sessions. In ab-initio mode, the protocol initializes preprocessing, metadata
            parsing, mask preparation, and neural network optimization from scratch. In
            continuation mode, previously generated weights and latent variables are loaded in
            order to extend or refine an existing model.

            The protocol also supports multi-body dynamics analysis. In this configuration,
            several masks defining independent structural regions can be provided. These masks
            allow the network to model correlated motions between rigid or semi-rigid domains,
            which is particularly useful for flexible macromolecular assemblies and large
            molecular complexes.

            Masking and Structural Focus

            Masking plays a central biological role in the protocol because it determines which
            structural regions contribute to network optimization. The protocol encourages the
            use of solvent masks to restrict learning to meaningful density regions while
            excluding solvent noise and empty image areas.

            For single-body training, one global mask is typically sufficient to isolate the
            molecular region of interest. In multi-body workflows, several masks can be defined
            to represent distinct structural domains. These masks are automatically organized
            and converted into STAR metadata files required by OPUS-DSD for rigid-body dynamics
            analysis.

            Biologically, careful mask design is essential. Stable structural cores generally
            produce more robust latent representations, while poorly defined masks may introduce
            noise or unstable dynamics into the learned conformational landscape.

            Latent Space Representation

            The protocol learns a latent representation of structural variability through the
            z-dimensional latent variable space. This latent space encodes continuous structural
            changes observed across particle images.

            Lower latent dimensions are generally recommended during exploratory analyses or
            low-resolution experiments, while higher-dimensional latent spaces may capture more
            complex motions at the cost of increased computational complexity and risk of
            overfitting. The protocol also supports an additional affine latent space for
            multi-body dynamics, allowing the network to describe relative body motions.

            From a biological perspective, latent variables should not be interpreted directly
            as physical coordinates but rather as abstract representations of conformational
            variability. Nevertheless, smooth trajectories in latent space frequently correlate
            with biologically meaningful structural transitions.

            Neural Network Architecture

            The protocol allows customization of both encoder and decoder architectures. Users
            can define the number of hidden layers and the number of nodes per layer for each
            network component.

            The encoder transforms particle images into latent representations, while the
            decoder reconstructs structural information from latent coordinates. Advanced users
            may optimize architecture depth and dimensionality depending on dataset complexity,
            particle count, and available GPU resources.

            The protocol also exposes several optimization hyperparameters, including batch
            size, learning rate, weight decay, beta regularization, UMAP-inspired restraints,
            and gradient accumulation. These parameters influence training stability,
            regularization strength, and convergence behavior.

            In biological datasets with strong flexibility or low signal-to-noise ratios,
            stronger regularization and larger reconstruction blur factors may improve training
            robustness. Conversely, highly homogeneous datasets may benefit from lighter
            regularization and smaller latent dimensions.

            Metadata Parsing and Training Preparation

            Before training begins, the protocol parses particle orientations and microscope
            information into OPUS-DSD-compatible formats. Pose metadata are extracted from STAR
            files, and CTF parameters are converted into serialized representations used during
            optimization.

            The protocol automatically corrects translation dimensionality inconsistencies in
            pose files to ensure compatibility with downstream OPUS-DSD routines. For
            multi-body training, additional STAR files describing body masks and rigid-body
            relationships are generated automatically.

            During continuation training, previously generated weights and latent embeddings are
            loaded to resume optimization without restarting from scratch. This enables
            progressive refinement of the latent space across multiple training sessions.

            GPU Execution and Computational Behavior

            Training is executed through GPU-enabled external OPUS-DSD programs. The protocol
            automatically configures thread-related environment variables and supports execution
            on multiple GPUs.

            Lazy loading can be enabled to optimize memory usage during training. When enabled,
            particles are loaded dynamically instead of remaining entirely in memory. This is
            particularly useful for large cryo-EM datasets processed on systems with limited
            GPU memory resources.

            Validation and Consistency Checks

            Several validation checks are performed before execution. The protocol verifies that
            particle box sizes are even, that projection alignments are available, and that the
            number of epochs is biologically and computationally meaningful.

            Template resolution values must also satisfy divisibility constraints required by
            convolutional network operations. Additional consistency checks are applied in
            multi-body workflows to ensure particles remain within valid dimensional limits for
            OPUS-DSD dynamics analysis.

            Outputs and Their Interpretation

            The protocol generates trained neural network weights, latent-space embeddings,
            parsed pose files, and CTF metadata required for downstream analysis. Outputs are
            reorganized automatically into epoch-specific result directories to simplify later
            evaluation and continuation training.

            Biologically, the trained latent space provides a compact representation of
            conformational variability across the dataset. These learned representations can be
            explored using dedicated analysis protocols to generate structural trajectories,
            cluster conformations, or reconstruct representative density maps.

            Practical Recommendations

            In most biological workflows, it is advisable to begin with moderate latent
            dimensions and conservative network architectures. Initial exploratory training
            often benefits from lower-dimensional embeddings before attempting more complex
            representations.

            Proper masking is one of the most important factors influencing successful training.
            Stable masks centered on biologically conserved regions generally improve latent
            space quality and convergence stability.

            Multi-body training should be reserved for datasets where independent domain motions
            are expected biologically. In these cases, carefully defining masks for each body
            substantially improves the interpretability of the resulting dynamics.

            Final Perspective

            OPUS-DSD training is not simply a neural network optimization procedure but a
            biologically driven framework for learning structural variability directly from
            cryo-EM particle images. Careful selection of latent dimensions, masking strategy,
            training continuation policies, and optimization parameters strongly influences the
            biological interpretability of the resulting conformational landscape.
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
                      label="Initial Training",
                      help="If preprocess data is required, set to yes, if training data is required, set to no.")

        form.addParam('inputMask', params.PointerParam, pointerClass='VolumeMask',
                      condition='abInitio==%s' % True, allowsNull=True,
                      label="Input Mask",
                      help="The suggestion is to use an already given solvent mask. "
                           "If it isn't given, it must be calculated from a volume separately, as it's necessary. "
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

        form.addParam('zDim', params.IntParam, default=8,
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

        form.addParam('betaControl', params.FloatParam, default=0.5,
                      condition='abInitio==%s' % True,
                      label='Beta restraint strength for KL target',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Beta parameter that controls the strength of the beta-VAE prior. The larger '
                           'the argument, the stronger the strength of the standard Gaussian restraint. The scale of '
                           'the beta-control should be proportional to the SNR of the dataset.')

        form.addParam('lamb', params.FloatParam, default=0.5,
                      condition='abInitio==%s' % True,
                      label='Restraint strength for umap prior',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='This controls the stretch of the UMAP-inspired prior for '
                           'the encoder network that encourages the encoding of structural '
                           'information for images in the same projection class. Possible values between [0.1, 3.].')

        form.addParam('bfactor', params.FloatParam, default=4.0,
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

        form.addParam('templateres', params.IntParam, default=144,
                      condition='abInitio==%s' % True,
                      label='Output size',
                      help='The output size of the reconstructed 3D volume in the intermediate steps of the convolutional network.'
                           ' You may keep it around > D, as it would mean a increase on the resolution. Problem: '
                           'the higher this value is, more memory will be consumed.')

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
            inMask = self._getInputMask().getFileName()
            shutil.copy(inMask, maskFilename)

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
        args = self._getTrainingArgs()

        if self.abInitio:
            self._runProgram('train_multi', args)
            self._outputRegroup(self.numEpochs.get() - 2)
        else:
            self._runProgram('train_multi', args[0])
            self._outputRegroup(args[1])

    # --------------------------- INFO functions ------------------------------

    def _summary(self):
        summary = ["Training CV for %d epochs." % self.numEpochs]

        return summary

    def _validate(self):
        errors = []

        if self._getBoxSize() % 2 != 0:
            errors.append("Box size must be even!")

        if not self._inputHasAlign():
            errors.append("Input particles have no alignment!")

        if self.numEpochs.get() < 2:
            errors.append("Number of epochs must be at least 2!")

        if self.templateres.get() % 16 != 0:
            errors.append("Template resolution (templateres) must be divisible by 16)!")

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
                args += '--downfrac 1.0 '
            else:
                raise ValueError("Error while asserting, please change the box size factor accordingly, as particles"
                                 "must remain in 128x128 in multibody dynamics. ")
        else:
            args += '--downfrac 1.0 '

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

    def _getInputMask(self):
        return self.inputMask.get()

    def _getBoxSize(self):
        return self._getInputParticles().getXDim()

    def _getExtra(self, *paths):
        return os.path.abspath(self._getExtraPath(*paths))

    def _getWorkDir(self):
        workDir = [dir for dir in os.listdir(self._getExtra()) if dir.startswith('Results')][0]
        return self._getExtra(workDir)

    def _runProgram(self, program, args):
        gpus = ','.join(str(i) for i in self.getGpuList())
        threads = f'{self.numberOfThreads.get()}'
        env = pwutils.Environ()
        env.update({
            'OMP_NUM_THREADS': threads,
            'MKL_NUM_THREADS': threads,
            'OPENBLAS_NUM_THREADS': threads,
            'NUMEXPR_NUM_THREADS': threads,
            'NUMBA_NUM_THREADS': threads
        })
        self.runJob(Plugin.getProgram(program, gpus, fromCryodrgn=True), args, env=env)

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