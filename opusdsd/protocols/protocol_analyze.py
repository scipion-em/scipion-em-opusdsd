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
import shutil
import pyworkflow.utils as pwutils
import pyworkflow.protocol.params as params
import pyworkflow.object as pwobj
from pyworkflow.constants import PROD
from pwem.protocols import ProtProcessParticles, ProtFlexBase
import pwem.objects as emobj
from ..constants import *
from ..utils import *

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
            'input_mask': self._getExtra('input_mask.mrc'),
            'output_vol': 'vol_%(id)01d.mrc',
            'output_volN_pc': 'pc%(PC)d/vol_%(id)01d.mrc',
            'output_volN_km': 'kmeans%(ksamples)d/vol_%(id)01d.mrc',
            'z_values': 'z_values.txt',
            'z_valuesN_pc': 'pc%(PC)d/z_pc.txt',
            'z_valuesN_km': 'kmeans%(ksamples)d/centers.txt'
        }
        self._updateFilenamesDict(myDict)

    # --------------------------- DEFINE param functions ----------------------

    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', params.PointerParam,
                      pointerClass="SetOfParticles, SetOfParticlesFlex",
                      label='Input Particles')

        form.addSection(label='Analysis')
        form.addParam('opusDSDTrainingProtocol', params.PointerParam, label="Opus-DSD trained network",
                      pointerClass='OpusDsdProtTrain',
                      help="Previously executed 'training - Opus-DSD'. "
                           "This will allow to load the results the network trained in that protocol "
                           "to be used during the analysis.")

        group = form.addGroup('Volume Generation')
        group.addParam('sampleMode', params.EnumParam,
                       choices=['KMEANS', 'PCA'], default=KMEANS,
                       label='Sample Mode', help='Selection of analysis method for volume generation')

        group.addParam('numPCs', params.IntParam, default=4, condition='sampleMode==%s' % PCA,
                       label='Number of principal components',
                       help='Number of principal components to sample for traversal.')

        group.addParam('PC', params.IntParam, default=1, condition='sampleMode==%s' % PCA,
                       label='PC', help='Specific principal component to choose zValues for volume generation')

        group.addParam('psamples', params.IntParam, default=12, condition='sampleMode==%s' % PCA,
                      label='Number of PC samples to generate',
                      help="When analyzing, this protocol uses the principal component "
                           "algorithm to analyze the latent space into "
                           "components, and generate a trajectory along the specific principle component. "
                           "The goal is to provide z values along PC for a subsequent"
                           "generation of volumes. psamples mod zDim must be 0.")

        group.addParam('ksamples', params.IntParam, default=24, condition='sampleMode==%s' % KMEANS,
                      label='Number of K-means samples to generate',
                      help="When analyzing, this protocol uses the k-means clustering "
                           "algorithm to partition the latent space into "
                           "regions, and generate a density map from the center of each of these "
                           "regions. The goal is to provide a tractable number "
                           "of representative density maps to visually inspect in a "
                           "subsequent generation of volumes. ksamples mod zDim must be 0.")

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

        self._insertFunctionStep(self.convertInputStep)
        self._insertFunctionStep(self.runAnalysisStep)
        self._insertFunctionStep(self.runEvalVolStep)
        self._insertFunctionStep(self.createOutputStep)

    # --------------------------- STEPS functions -----------------------------

    def convertInputStep(self):
        self.initEpoch = os.path.basename(self._getWorkDir()).split('.')[1]
        self.zDim = self._getOpusDSDTrainingProtocol().zDim
        self.downFrac = self._getOpusDSDTrainingProtocol().downFrac
        self.inputMask = self._getFileName('input_mask')

        self.weights = self._getWorkDir() + f'/weights.{self.initEpoch}.pkl'
        self.weightsNew = self._getWorkDir() + f'/weights_new.{self.initEpoch}.pkl'
        self.config = self._getExtra() + '/config.pkl'
        self.runJob(Plugin.getTorchLoadProgram(self._getWorkDir(), self.weights, self.weightsNew, 'weights'), '')

        # When computing the encoder, we need a new Apix for asserting equal shapes on convolutional matrices
        self.wr = window_r(self.inputMask)
        crop_vol_size = self._getWorkDir() + '/crop_vol_size'
        self.runJob(Plugin.getTorchLoadProgram(self._getWorkDir(), self.weightsNew, crop_vol_size, 'eval_vol'), '')
        self.crop_vol_size = np.loadtxt(crop_vol_size + '.txt').shape[-1]

        render_size = (int(float(self._getBoxSize()) * float(self.downFrac)) // 2) * 2
        newApix = self._getInputParticles().getSamplingRate() * self._getBoxSize() / render_size
        self.newApix = newApix * float(self.crop_vol_size) / (float(self._getBoxSize()) * float(self.downFrac) * self.wr)

    def runAnalysisStep(self):
        """ Call OPUS-DSD with the appropriate parameters to analyze """
        args = self._getAnalyzeArgs()
        self._runProgram('analyze', args)

    def runEvalVolStep(self):
        """ Call OPUS-DSD with the appropriate parameters to analyze """
        args = self._getEvalVolArgs()
        self._runProgram('eval_vol', args)

    def createOutputStep(self):
        """ Create the protocol outputs. """
        # Creating a set of particles with z_values
        inSet = self._getInputParticles()
        zIterValues = iter(self._getParticlesZvalues(self.initEpoch))

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

        if os.path.exists(self.weightsNew):
            outSet.getFlexInfo().setAttr(WEIGHTS, pwobj.String(self.weightsNew))
        else:
            outSet.getFlexInfo().setAttr(WEIGHTS, pwobj.String(self.weights))

        outSet.getFlexInfo().setAttr(CONFIG, pwobj.String(self.config))
        outSet.getFlexInfo().setAttr(ZDIM, pwobj.Integer(self.zDim))
        outSet.getFlexInfo().setAttr(DOWNFRAC, pwobj.Float(self.downFrac))
        outSet.getFlexInfo().setAttr(CROP_VOL_SIZE, pwobj.Integer(self.crop_vol_size))
        outSet.getFlexInfo().setAttr(WINDOW_R, pwobj.Float(self.wr))

        self._defineOutputs(outputParticles=outSet)
        self._defineSourceRelation(inSet, outSet)

        # Creating a set of volumes with z_values depending on the sampleMode
        fn = self._getExtra('volumes.sqlite')
        files, zValues = self._getVolumesZCalc(sampleMode=self.sampleMode.get(), initEpoch=self.initEpoch, zDim=self.zDim)
        volSet = self._createVolumeZSet(files, zValues, fn, round(self.newApix, 2))

        self._defineOutputs(outputVolumes=volSet)
        self._defineSourceRelation(inSet, volSet)

    # --------------------------- INFO functions ------------------------------

    def _summary(self):
        initEpoch = os.path.basename(self._getWorkDir()).split('.')[1]
        zDim = self._getOpusDSDTrainingProtocol().zDim

        summary = ["Analyzing results for the %d epoch." % int(initEpoch)]
        if self.sampleMode.get() == PCA:
            summary += ["Number of volumes to generate when PCA is running is of a total of "
                        f"{int(4 * self.psamples.get() / int(zDim))}."]
        elif self.sampleMode.get() == KMEANS:
            summary += ["Number of volumes to generate when KMEANS is running is of a total of "
                        f"{int(4 * self.ksamples.get() / int(zDim))}."]

        return summary

    def _validateBase(self):
        errors = []

        return errors

    # --------------------------- ARGS functions -----------------------------

    def _getAnalyzeArgs(self):
        poseDir = self._getWorkDir() + f'/pose.{self.initEpoch}.pkl'

        args = self._getWorkDir()
        args += ' %d ' % int(self.initEpoch)
        args += '--outdir %s ' % self._out(self.initEpoch)
        args += '--vanilla '
        args += '--D %d ' % self._getOpusDSDTrainingProtocol().templateres
        args += '--pose %s ' % poseDir
        args += '--pc %d ' % self.numPCs

        if self.ksamples.get() % int(self.zDim) == 0:
            args += '--ksample %d ' % self.ksamples
        else:
            raise ValueError(
                f"Error while asserting, ksamples mod zDim {self.zDim} (selected in previous training) must be 0, "
                "please change ksamples accordingly")

        if self.psamples.get() % int(self.zDim) == 0:
            args += '--psample %d' % self.psamples
        else:
            raise ValueError(
                f"Error while asserting, psamples mod zDim {self.zDim} (selected in previous training) must be 0, "
                "please change psamples accordingly")

        return args

    def _getEvalVolArgs(self):
        if self.sampleMode.get() == PCA:
            zFile = self._out(self.initEpoch, self._getFileName('z_valuesN_pc', PC=self.PC.get()))
        elif self.sampleMode.get() == KMEANS:
            zFile = self._out(self.initEpoch, self._getFileName('z_valuesN_km', ksamples=self.ksamples.get()))

        if os.path.exists(self.weightsNew):
            args = '--load %s ' % self.weightsNew
        else:
            args = '--load $s ' % self.weights

        args += '--config %s ' % self.config

        if self.sampleMode.get() == KMEANS:
            args += '-o %s ' % self._out(self.initEpoch, 'kmeans%d') % self.ksamples.get()
        elif self.sampleMode.get() == PCA:
            args += '-o %s ' % self._out(self.initEpoch, 'pc%d') % self.PC.get()

        args += '--prefix vol_ '
        args += '--zfile %s ' % zFile
        args += '--Apix %f ' % round(self.newApix, 2)
        args += '--enc-layers %d ' % self._getOpusDSDTrainingProtocol().qLayers
        args += '--enc-dim %d ' % self._getOpusDSDTrainingProtocol().qDim
        args += '--zdim %d ' % int(self.zDim)
        args += '--encode-mode grad '
        args += '--dec-layers %d ' % self._getOpusDSDTrainingProtocol().pLayers
        args += '--dec-dim %d ' % self._getOpusDSDTrainingProtocol().pDim
        args += '--pe-type vanilla '
        args += '--template-type conv '
        args += '--activation relu'

        return args

    # --------------------------- UTILS functions -----------------------------

    def _getInputParticles(self):
        return self.inputParticles.get()

    def _getBoxSize(self):
        return self._getInputParticles().getXDim()

    def _getExtra(self, *paths):
        return os.path.abspath(self._getExtraPath(*paths))

    def _runProgram(self, program, args, fromXmipp=False):
        gpus = ','.join(str(i) for i in self.getGpuList())
        if not fromXmipp:
            self.runJob(Plugin.getProgram(program, gpus, fromCryodrgn=True), args, env=pwutils.Environ())
        else:
            self.runJob(Plugin.getXmippProgram(program), args)

    def _getWorkDir(self):
        workDir = [dir for dir in os.listdir(self._getExtra()) if dir.startswith('Results')][0]
        return self._getExtra(workDir)

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
        self.runJob(Plugin.getTorchLoadProgram(self._getWorkDir(), zEpochFile, zEpochFileNew, 'pkl'), '')

    def _getParticlesZvalues(self, initEpoch):
        """
        Read from z.npz file the particles z_values
        """
        zEpochFileNpz = self._getWorkDir() + '/output_file.npz'
        self._setParticlesZValues(initEpoch)
        zValues = np.load(zEpochFileNpz)
        return zValues['mu']

    def _getVolumeZvalues(self, zValueFile):
        """
        Read from z_values.txt file the volume z_values
        """
        return np.loadtxt(zValueFile, dtype=float).tolist()

    def _getVolumesZCalc(self, sampleMode, initEpoch, zDim):
        """ Returns a list of volume names and their zValues. """
        vols = []
        zValues = []
        if self._hasMultLatentVars():
            if sampleMode == KMEANS:
                fn = 'output_volN_km'
                ksamples = self.ksamples.get()
                zValue = 'z_valuesN_km'
                zValues.extend(self._getVolumeZvalues(self._out(initEpoch, self._getFileName(zValue, ksamples=ksamples))))
                for volId in range(int(4 * ksamples / int(zDim))):
                    volFn = self._out(initEpoch, self._getFileName(fn, ksamples=ksamples, id=volId))
                    vols = self._appendVolumes(vols, volFn)

            elif sampleMode == PCA:
                fn = 'output_volN_pc'
                zValue = 'z_valuesN_pc'
                PC = self.PC.get()
                psamples = self.psamples.get()
                zValues.extend(self._getVolumeZvalues(self._out(initEpoch, self._getFileName(zValue, PC=PC))))
                for volId in range(int(4 * psamples / int(zDim))):
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
        return int(self._getOpusDSDTrainingProtocol().zDim) > 1

    def _appendVolumes(self, vols, volFn):
        if os.path.exists(volFn):
             vols.append(volFn)
        else:
            raise FileNotFoundError("Volume %s does not exists. Please select a valid epoch number." % volFn)
        return vols

    def _getOpusDSDTrainingProtocol(self):
        return self.opusDSDTrainingProtocol.get()