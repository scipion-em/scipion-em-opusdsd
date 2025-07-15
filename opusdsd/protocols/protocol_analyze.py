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
import os, shutil, glob
from email.policy import default

import numpy as np
import pyworkflow.utils as pwutils
import pyworkflow.protocol.params as params
import pyworkflow.object as pwobj
from pyparsing import conditionAsParseAction
from pyworkflow.constants import PROD

from pwem.protocols import ProtProcessParticles, ProtFlexBase
import pwem.objects as emobj

from .. import Plugin
from ..constants import *

class OpusDsdProtAnalyze(ProtProcessParticles,ProtFlexBase):
    """
    Protocol to analyze results from OPUS-DSD neural network.
    """
    _label = 'opusdsd analyze'
    _devStatus = PROD

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _createFilenameTemplatesAnalyze(self):
        """ Centralize how files are called within the analysis protocol. """
        myDict = {
            'workTrainDir': self._getOpusDSDTrainingProtocol()._getExtra(),
            'output_vol': 'vol_%(id)01d.mrc',
            'output_volN_pc': 'pc%(PC)d/vol_%(id)01d.mrc',
            'output_volN_km': 'kmeans%(ksamples)d/vol_%(id)01d.mrc',
            'z_values': 'z_values.txt',
            'z_valuesN_pc': 'pc%(PC)d/z_values.txt',
            'z_N_pc': 'pc%(PC)d/z_pc.txt',
            'z_valuesN_km': 'kmeans%(ksamples)d/z_values.txt'
        }
        self._updateFilenamesDict(myDict)

    # --------------------------- DEFINE param functions ----------------------

    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', params.PointerParam,
                      pointerClass="SetOfParticles, SetOfParticlesFlex",
                      label='Input Particles')

        form.addParam('Apix', params.FloatParam, default=-1,
                      label='Pixel size in A/pix',
                      help='If left as -1, pixel size will be the same as the sampling rate of the input particles.')

        form.addParam('boxSize', params.IntParam, default=-1,
                      label='Box size of reconstruction (pixels)',
                      help='If left as -1, box size will be the same as the dimensions of the input particles.')

        group = form.addGroup('Volume Generation')
        group.addParam('sampleMode', params.EnumParam,
                       choices=['KMEANS', 'PCS'], default=PCS,
                       label='Sample Mode', help='Selection of analysis method for volumen generation')

        group.addParam('downSampling', params.BooleanParam, default=False, expertLevel=params.LEVEL_ADVANCED,
                       label='DownSampling is needed?', help='Downsample volumes to this box size (pixels).')

        group.addParam('downSample', params.IntParam, condition='downSampling', expertLevel=params.LEVEL_ADVANCED,
                       label='DownSampling factor', help='DownSampling factor in case you want to resize volumes.')

        group.addParam('PC', params.IntParam, default=1,
                       label='PC', help='Specific principal component to choose zValues for volume generation')

        form.addSection(label='Analysis')
        form.addParam('opusDSDTrainingProtocol', params.PointerParam, label="Opus-DSD trained network",
                      pointerClass='OpusDsdProtTrain',
                      help="Previously executed 'training - Opus-DSD'. "
                           "This will allow to load the results the network trained in that protocol "
                           "to be used during the analysis.")

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

        form.addParam('zDim', params.IntParam, default=12,
                      validators=[params.Positive],
                      label='Dimension of latent variable',
                      help='It is recommended to first train on lower '
                           'resolution images (e.g. D=128) with '
                           '--zdim 1 and with --zdim 12 using the '
                           'default architecture (fast). Values between [1, 12].')

        group = form.addGroup('PC Analysis', expertLevel=params.LEVEL_ADVANCED)
        group.addParam('numPCs', params.IntParam, default=4,
                      label='Number of principal components', expertLevel=params.LEVEL_ADVANCED,
                      help='Number of principal components to sample for traversal.')

        group.addParam('psamples', params.IntParam, default=10,
                      label='Number of PC samples to generate', expertLevel=params.LEVEL_ADVANCED,
                      help="*cryodrgn analyze* uses the principal component "
                           "algorithm to analyze the latent space into "
                           "components (by default p=10 components), and generate a "
                           "trajectory along the specific principle component. "
                           "The goal is to provide z values along PC for a subsequent"
                           "generation of volumes. ")

        group = form.addGroup('K-Means Clustering', expertLevel=params.LEVEL_ADVANCED)
        group.addParam('ksamples', params.IntParam, default=10,
                      label='Number of K-means samples to generate', expertLevel=params.LEVEL_ADVANCED,
                      help="*cryodrgn analyze* uses the k-means clustering "
                           "algorithm to partition the latent space into "
                           "regions (by default k=20 regions), and generate a "
                           "density map from the center of each of these "
                           "regions. The goal is to provide a tractable number "
                           "of representative density maps to visually inspect in a "
                           "subsequent generation of volumes. ")

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

        form.addSection(label='Advanced')
        form.addParam('skipVol', params.BooleanParam, default=True,
                      label='Skip Volume', expertLevel=params.LEVEL_ADVANCED,
                      help='Skip generation of volumes.')

        form.addParam('skipUMAP', params.BooleanParam, default=False,
                      label='Skip UMAP', expertLevel=params.LEVEL_ADVANCED,
                      help='Skip running UMAP on latents.')

        form.addParam('imageSize', params.IntParam, default=224,
                      label='Image Size', expertLevel=params.LEVEL_ADVANCED,
                      help='Set the size of image.')

        form.addParam('deform', params.BooleanParam, default=False,
                      label="Use zTemplate?", expertLevel=params.LEVEL_ADVANCED)

        form.addParam('zTemplate', params.StringParam,
                      condition='deform', label="zTemplate", expertLevel=params.LEVEL_ADVANCED,
                      help="Path for template encoding when deforming the structure is required.")

        form.addParam('zTemplateIndex', params.IntParam, default=20,
                      condition='deform', label="zTemplate Index", expertLevel=params.LEVEL_ADVANCED,
                      help="Path for template encoding when deforming the structure is required.")

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
        self._createFilenameTemplatesAnalyze()

        pwutils.cleanPath(self._getExtra())
        shutil.copytree(self._getFileName('workTrainDir'), self._getExtra())
        shutil.move(self._getExtra() + '/run.log', self._getWorkDir() + '/run.log')
        initEpoch = os.path.basename(self._getWorkDir()).split('.')[1]

        self._insertFunctionStep(self.runAnalysisStep, initEpoch)
        self._insertFunctionStep(self.runEvalVolStep, initEpoch)
        self._insertFunctionStep(self.createOutputStep, initEpoch)

    # --------------------------- STEPS functions -----------------------------

    def runAnalysisStep(self, initEpoch):
        """ Call OPUS-DSD with the appropriate parameters to analyze """
        poseDir = self._getWorkDir() + f'/pose.{initEpoch}.pkl'

        args = self._getWorkDir()
        args += ' %d ' % int(initEpoch)
        args += '--outdir %s ' % self._out(initEpoch)

        if self.skipVol:
            args += '--skip-vol '

        if self.skipUMAP:
            args += '--skip-umap '

        args += '--vanilla '
        args += '--D %d ' % self.imageSize
        args += '--pose %s ' % poseDir

        if self.Apix.get() != -1:
            args += '--Apix %f ' % self.Apix.get()
        else:
            args += '--Apix %f ' % self._getInputParticles().getSamplingRate()

        args += '--pc %d ' % self.numPCs
        args += '--ksample %d ' % self.ksamples
        args += '--psample %d' % self.psamples

        self._runProgram('analyze', args)

    def runEvalVolStep(self, initEpoch):
        """ Call OPUS-DSD with the appropriate parameters to analyze """
        config = self._getExtra() + '/config.pkl'
        zFile = self._out(initEpoch, self._getFileName('z_N_pc', PC=self.PC.get()))

        args = '--config %s ' % config

        if self.sampleMode.get() == KMEANS:
            args += '-o %s ' % self._out(initEpoch, 'kmeans%d') % self.ksamples.get()
        elif self.sampleMode.get() == PCS:
            args += '-o %s ' % self._out(initEpoch, 'pc%d') % self.PC.get()

        args += '--prefix vol_ '

        if self.skipVol:
            args += '--zfile %s ' % zFile

        if self.deform:
            args += '--deform '
            args += '--template-z %s ' % self.zTemplate
            args += '--template-z-ind %d ' % self.zTemplateIndex

        args += '--num-bodies 0 '

        if self.Apix.get() != -1:
            args += '--Apix %f ' % self.Apix.get()
        else:
            args += '--Apix %f ' % self._getInputParticles().getSamplingRate()

        if self.downSampling:
            args += '--downsample %d ' % self.downSample

        if self.boxSize.get() != -1:
            args += '-D %d ' % self.boxSize.get()
        else:
            args += '-D %d ' % self._getBoxSize()

        args += '--enc-layers %d ' % self.qLayers
        args += '--enc-dim %d ' % self.qDim

        if len(np.loadtxt(zFile)) % self.zDim.get() != 0:
            self.zDim = self._fixZDim(zFile, self.zDim.get())

        args += '--zdim %d ' % self.zDim
        args += '--encode-mode grad '
        args += '--dec-layers %d ' % self.pLayers
        args += '--dec-dim %d ' % self.pDim
        args += '--pe-type vanilla '
        args += '--template-type conv '
        args += '--activation relu'

        self._runProgram('eval_vol', args)

    def createOutputStep(self, initEpoch):
        """ Create the protocol outputs. """
        weights = self._getWorkDir() + f'/weights.{initEpoch}.pkl'
        config = self._getExtra() + '/config.pkl'

        # Creating a set of particles with z_values
        inSet = self._getInputParticles()
        zIterValues = iter(self._getParticlesZvalues(initEpoch))

        outSet = self._createSetOfParticlesFlex(progName=OPUSDSD)
        outSet.copyInfo(inSet)
        outSet.setHasCTF(inSet.hasCTF())
        outSet.getFlexInfo().setProgName(OPUSDSD)

        for particle, zValue in zip(inSet, zIterValues):
            outParticle = emobj.ParticleFlex(progName=OPUSDSD)
            outParticle.copyInfo(particle)
            outParticle.getFlexInfo().setProgName(OPUSDSD)
            outParticle.setZFlex(list(zValue))
            outSet.append(outParticle)

        outSet.getFlexInfo().setAttr(WEIGHTS, pwobj.String(weights))
        outSet.getFlexInfo().setAttr(CONFIG, pwobj.String(config))

        self._defineOutputs(outputParticles=outSet)
        self._defineSourceRelation(inSet, outSet)

        # Creating a set of volumes with z_values depending on the sampleMode
        fn = self._getExtra('volumes.sqlite')
        files, zValues = self._getVolumesZCalc(sampleMode=PCS, initEpoch=initEpoch)
        if self.Apix.get() != -1:
            volSet = self._createVolumeZSet(files, zValues, fn,
                                            params.Float(self.Apix.get()))
        else:
            volSet = self._createVolumeZSet(files, zValues, fn,
                                            params.Float(self._getInputParticles().getSamplingRate()))

        self._defineOutputs(outputVolumes=volSet)
        self._defineSourceRelation(inSet, volSet)

    # --------------------------- INFO functions ------------------------------

    def _summary(self):
        summary = ["Analyzing results for %d epochs." % self.numEpochs]

        return summary

    def _validateBase(self):
        errors = []

        return errors

    # --------------------------- UTILS functions -----------------------------

    def _getInputParticles(self):
        return self.inputParticles.get()

    def _getBoxSize(self):
        return self._getInputParticles().getXDim()

    def _getExtra(self, *paths):
        return os.path.abspath(self._getExtraPath(*paths))

    def _runProgram(self, program, args, fromCryodrgn=True):
        gpus = ','.join(str(i) for i in self.getGpuList())
        self.runJob(Plugin.getProgram(program, gpus, fromCryodrgn=fromCryodrgn), args)

    def _getWorkDir(self):
        workDir = [dir for dir in os.listdir(self._getExtra()) if dir.startswith('CV')]
        return self._getExtra(workDir[0])

    def _out(self, initEpoch, *p):
        if self._hasMultLatentVars():
            return os.path.join(self._getWorkDir() + f'/defanalyze.{initEpoch}', *p)
        else:
            return os.path.join(self._getWorkDir() + f'/analyze.{initEpoch}', *p)

    def _setParticlesZValues(self, initEpoch):
        """
        Read from z.pkl file the particles z_values and turns it into a z.npz file
        """
        zEpochFile = self._getWorkDir() + f'/z.{initEpoch}.pkl'
        zEpochFileNew = self._getWorkDir() + '/output_file'
        self.runJob(Plugin.getTorchLoadProgram(self._getWorkDir(), zEpochFile, zEpochFileNew), '')

    def _getParticlesZvalues(self, initEpoch):
        """
        Read from z.npz file the particles z_values
        :return: a numpy array with the particles z_values
        """
        zEpochFileNpz = self._getWorkDir() + '/output_file.npz'
        self._setParticlesZValues(initEpoch)
        zValues = np.load(zEpochFileNpz)
        return zValues['mu']

    def _getVolumeZvalues(self, zValueFile):
        """
        Read from z_values.txt file the volume z_values
        :return: a list with the volumes z_values
        """
        return np.loadtxt(zValueFile, dtype=float).tolist()

    def _getVolumesZCalc(self, sampleMode, initEpoch):
        """ Returns a list of volume names and their zValues. """
        vols = []
        zValues = []
        if not self.skipVol:
            if self._hasMultLatentVars():
                if sampleMode == KMEANS:
                    fn = 'output_volN_km'
                    ksamples = self.ksamples.get()
                    zValue = 'z_valuesN_km'
                    zValues.extend(self._getVolumeZvalues(self._out(initEpoch, self._getFileName(zValue, ksamples=ksamples))))
                    for volId in range(ksamples):
                        volFn = self._out(initEpoch, self._getFileName(fn, ksamples=ksamples, id=volId))
                        vols = self._appendVolumes(vols, volFn)

                else:
                    fn = 'output_volN_pc'
                    zValue = 'z_valuesN_pc'
                    PC = self.PC.get()
                    numPCs = self.numPCs.get()
                    zValues.extend(self._getVolumeZvalues(self._out(initEpoch, self._getFileName(zValue, PC=PC))))
                    for volId in range(numPCs):
                        volFn = self._out(initEpoch, self._getFileName(fn, PC=PC, id=volId))
                        vols = self._appendVolumes(vols, volFn)
            else:
                fn = 'output_vol'
                zValue = 'z_values'
                zValues.extend(self._getVolumeZvalues(self._out(initEpoch, self._getFileName(zValue))))
                volFn = self._out(initEpoch, self._getFileName(fn, id=0))
                vols = self._appendVolumes(vols, volFn)

        else:
            if self._hasMultLatentVars():
                if sampleMode == KMEANS:
                    print('ERROR: z_values for KMEANS sample mode were not generated. Skipping to PC sample mode...')
                    fn = 'output_volN_pc'
                    zValue = 'z_N_pc'
                    PC = self.PC.get()
                    numPCs = self.numPCs.get()
                    zValues.extend(self._getVolumeZvalues(self._out(initEpoch, self._getFileName(zValue, PC=PC))))
                    for volId in range(numPCs):
                        volFn = self._out(initEpoch, self._getFileName(fn, PC=PC, id=volId))
                        vols = self._appendVolumes(vols, volFn)

                else:
                    fn = 'output_volN_pc'
                    zValue = 'z_N_pc'
                    PC = self.PC.get()
                    numPCs = self.numPCs.get()
                    zValues.extend(self._getVolumeZvalues(self._out(initEpoch, self._getFileName(zValue, PC=PC))))
                    for volId in range(numPCs):
                        volFn = self._out(initEpoch, self._getFileName(fn, PC=PC, id=volId))
                        vols = self._appendVolumes(vols, volFn)
            else:
                fn = 'output_vol'
                zValue = 'z_values'
                zValues.extend(self._getVolumeZvalues(self._out(initEpoch, self._getFileName(zValue))))
                volFn = self._out(initEpoch, self._getFileName(fn, id=0))
                vols = self._appendVolumes(vols, volFn)
        return vols, zValues

    def _createVolumeZSet(self, files, zValues, path, samplingRate, updateItemCallback=None):
        """
        Create a set of volume with the associated z_values
        :param files: list of the volumes path
        :param zValues: list with the volumes z_values
        :param path: output path
        :param samplingRate: volumes sampling rate
        :return: a set of volumes
        """
        pwutils.cleanPath(path)
        volSet = emobj.SetOfVolumes(filename=path)
        volSet.setSamplingRate(samplingRate)
        volId = 0
        if type(zValues[0]) is not list:
            # csvList requires each item as a list
            zValues = [[i] for i in zValues]
        for volFn in files:
            vol = emobj.Volume()
            vol.setFileName(volFn)
            vector = pwobj.CsvList()
            # We assume that each row "i" of z_values corresponds to each volumes with ID "i"
            volZValues = zValues[volId]
            vector._convertValue(volZValues)
            # Creating a new column in the volumes with the z_value
            setattr(vol, Z_VALUES, vector)
            if updateItemCallback:
                updateItemCallback(vol)
            volSet.append(vol)
            volId += 1
        return volSet

    def _hasMultLatentVars(self):
        return self.zDim > 1

    def _appendVolumes(self, vols, volFn):
        if os.path.exists(volFn):
             vols.append(volFn)
        else:
            raise FileNotFoundError("Volume %s does not exists. Please select a valid epoch number." % volFn)
        return vols

    def _getOpusDSDTrainingProtocol(self):
        return self.opusDSDTrainingProtocol.get()

    def _fixZDim(self, zFile, zDim):
        # In case zDim isn't a multiple of z size, we correct it so that its reshaping works.
        zSize = len(np.loadtxt(zFile))
        while zSize % zDim != 0:
            zDim = zDim - 1
        return zDim