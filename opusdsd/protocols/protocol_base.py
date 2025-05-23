# **************************************************************************
# *
# * Authors:     Grigory Sharov (gsharov@mrc-lmb.cam.ac.uk) [1]
# *              Yunior C. Fonseca Reyna (cfonseca@cnb.csic.es) [2]
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
import pickle
import numpy as np
import re
from glob import glob

from pwem.constants import ALIGN_PROJ, ALIGN_NONE
import pyworkflow.utils as pwutils
import pyworkflow.protocol.params as params
import pyworkflow.object as pwobj
from pyworkflow.plugin import Domain

from pwem.protocols import ProtProcessParticles
import pwem.objects as emobj

from .. import Plugin
from ..constants import (EPOCH_LAST, EPOCH_SELECTION, WEIGHTS, CONFIG, 
                         Z_VALUES)

KMEANS = 0
PCS = 1

convert = Domain.importFromPlugin('relion.convert', doRaise=True)

class OpusDsdProtBase(ProtProcessParticles):
    _label = None

    def _createFilenameTemplates(self):
        """ Centralize how files are called within the protocol. """
        def out(*p):
            return os.path.join(self.getOutputDir(f'analyze.{self._epoch}'), *p)

        myDict = {
            'output_vol': out('vol_%(id)03d.mrc'),
            'output_volN_km': out('kmeans%(ksamples)d/reference%(id)d.mrc'),
            'output_volN_pc': out('pc%(i)d/reference%(id)d.mrc'),
            'z_values': out('z_values.txt'),
            'z_valuesN_km': out('kmeans%(ksamples)d/centers.txt'),
            'z_valuesN_pc': out('pc%(i)d/z_pc.txt'),
            'weights': self.getOutputDir(f'weights.{self._epoch}.pkl'),
            'config': self.getOutputDir('config.pkl'),
            'input_parts': self._getExtraPath('input_particles.star'),
            'output_folder': out(),
            'output_poses': self.getOutputDir('poses.pkl'),
            'output_ctfs': self.getOutputDir('ctfs.pkl'),
        }

        self._updateFilenamesDict(myDict)

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addHidden('doInvert', params.BooleanParam, default=True)
        form.addSection(label='Input')
        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       label="Choose GPU IDs",
                       help="GPU may have several cores. Set it to zero"
                            " if you do not know what we are talking about."
                            " First core index is 0, second 1 and so on."
                            " You can use multiple GPUs - in that case"
                            " set to i.e. *0 1 2*.")
        
        form.addParam('doTraining', params.BooleanParam, default=True,
                      expertLevel=params.LEVEL_ADVANCED,
                      label="Do full training?",
                      help='The alternative is to analyze a previous results from a previous training job')
        
        form.addParam('relion31', params.BooleanParam, default=True,
                      expertLevel=params.LEVEL_ADVANCED,
                      label="Are particles from RELION 3.1 or later?")

        form.addParam('inputParticles', params.PointerParam,
                      pointerClass="SetOfParticles",
                      label='OPUS-DSD particles')

        form.addParam('zDim', params.IntParam, default=12,
                      validators=[params.Positive],
                      label='Dimension of latent variable',
                      help='It is recommended to first train on lower '
                           'resolution images (e.g. D=128) with '
                           '--zdim 1 and with --zdim 10 using the '
                           'default architecture (fast).')
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

        self._defineAdvancedParams(form)

        form.addSection(label='Analysis')
        form.addParam('viewEpoch', params.EnumParam,
                      choices=['last', 'selection'], default=EPOCH_LAST,
                      display=params.EnumParam.DISPLAY_LIST,
                      label="Epoch to analyze")

        form.addParam('epochNum', params.IntParam,
                      condition='viewEpoch==%d' % EPOCH_SELECTION,
                      label="Epoch number")
        
        form.addParam('numPCs', params.IntParam, default=4,
                      label='Number of principal components',
                      help='Number of principal components to sample for traversal')

        form.addParam('ksamples', params.IntParam, default=20,
                      label='Number of K-means samples to generate',
                      help="*cryodrgn analyze* uses the k-means clustering "
                           "algorithm to partition the latent space into "
                           "regions (by default k=20 regions), and generate a "
                           "density map from the center of each of these "
                           "regions. The goal is to provide a tractable number "
                           "of representative density maps to visually inspect. ")
        
        form.addParam('outApix', params.FloatParam, default=-1, 
                      label='Pixel size in A for output volumes',
                      help='If left at -1, the default behaviour is to use the input sampling rate')  

        form.addParallelSection(threads=16, mpi=0)

    def _defineAdvancedParams(self, form):
        """ Should be defined in subclasses. """
        raise NotImplementedError

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        if self.viewEpoch == EPOCH_LAST:
            self._epoch = self.numEpochs.get() - 1
        else:
            self._epoch = self.epochNum.get()
        self._createFilenameTemplates()

        if self.doTraining:
            self._insertFunctionStep(self.convertInputStep)
            self._insertFunctionStep(self.runParseMdStep)
            self._insertFunctionStep(self.runTrainingStep)
            self._insertFunctionStep(self.runAnalysisStep, self._epoch)

        self._insertFunctionStep(self.runEvalVolumesStep, self._epoch)
        #self._insertFunctionStep(self.runEvalPartsStep, self._epoch)
        self._insertFunctionStep(self.createOutputStep)

    # --------------------------- STEPS functions -----------------------------
    def convertInputStep(self):
        """ Create a star file as expected by OPUS-DSD."""
        outputFolder = self._getFileName('output_folder')
        pwutils.cleanPath(outputFolder)
        pwutils.makePath(outputFolder)

        imgSet = self.inputParticles.get()
        # Create links to binary files and write the relion .star file
        alignType = ALIGN_PROJ if self._inputHasAlign() else ALIGN_NONE
        starFilename = self._getFileName('input_parts')
        convert.writeSetOfParticles(
            imgSet, starFilename,
            outputDir=self._getExtraPath(), 
            alignType=alignType)
        
    def runParseMdStep(self):
        self._runProgram('parse_pose_star', self._getParsePosesArgs())
        self._runProgram('parse_ctf_star', self._getParseCtfArgs())

    def _getParsePosesArgs(self):
        args = ['%s' % self._getFileName('input_parts'),
                '-o %s' % self._getFileName('output_poses'),
                '-D %s' % self._getBoxSize(),
                '--Apix %s' % self.inputParticles.get().getSamplingRate()]
        
        if self.relion31:
            args.append('--relion31')
        return args

    def _getParseCtfArgs(self):
        acquisition = self.inputParticles.get().getAcquisition()
        args = ['%s' % self._getFileName('input_parts'),
                '-o %s' % self._getFileName('output_ctfs'),
                '-D %s' % self._getBoxSize(),
                '--Apix %s' % self.inputParticles.get().getSamplingRate(),
                '--kv %s' % acquisition.getVoltage(),
                '--cs %s' % acquisition.getSphericalAberration(),
                '-w %s' % acquisition.getAmplitudeContrast(),
                '--ps 0']  # required due to OPUS-DSD parsing bug

        if self.relion31:
            args.append('--relion31')
        return args

    def _getInputParticles(self):
        return self.inputParticles.get()

    def _getBoxSize(self):
        return self._getInputParticles().getXDim()

    def runTrainingStep(self):
        """ Should be implemented in subclasses. """
        raise NotImplementedError

    def runAnalysisStep(self, epoch):
        """ Run analysis step.
        Args:
            epoch: epoch number to be analyzed.
        """
        self._runProgram('analyze', self._getAnalyzeArgs(epoch), 
                         fromCryodrgn=False)
        
    def runEvalVolumesStep(self, epoch):
        """ Run eval vol step.
        Args:
            epoch: epoch number to be analyzed.
        """
        # eval vol for Kmeans then each PC
        self._runProgram('eval_vol', self._getEvalVolArgs(epoch), 
                        fromCryodrgn=False)
    
        # for i in range(1, self.numPCs.get()+1):
        #     self._runProgram('eval_vol', self._getEvalVolArgs(epoch, i, 
        #                                                       sampleMode=PCS), 
        #                     fromCryodrgn=False)

    def runEvalPartsStep(self, epoch):
        """ Run eval parts step.
        Args:
            epoch: epoch number to be analyzed.
        """
        # eval particles for Kmeans then each PC
        self._runProgram('parse_pose', self._getEvalVolArgs(epoch), 
                        fromCryodrgn=False)

        for i in range(1, self.numPCs.get()+1):
            self._runProgram('parse_pose', self._getEvalVolArgs(epoch, i, 
                                                                sampleMode=PCS), 
                             fromCryodrgn=False)

    def createOutputStep(self):
        """ Create the protocol outputs. """
        # Creating a set of particles with z_values
        #outImgSet = self._createParticleSet()
        #self._defineOutputs(Particles=outImgSet)

        # Creating a set of volumes with z_values for first Kmeans and then each PC
        fn = self._getExtraPath('volumes.sqlite')
        files, zValues = self._getVolumes()
        files, zValues = self._getVolumes(sampleMode=PCS, vols=files, zValues=zValues)
        setOfVolumes = self._createVolumeSet(files, zValues, fn, self.samplingRate)
        self._defineOutputs(Volumes=setOfVolumes)
        self._defineSourceRelation(self.inputParticles.get(), setOfVolumes)

    # --------------------------- INFO functions ------------------------------
    def _validateBase(self):
        errors = []

        if self.viewEpoch == EPOCH_SELECTION:
            ep = self.epochNum.get()
            total = self.numEpochs.get()
            if ep > total:
                errors.append(f"You can analyse only epochs 1-{total}")

        if self._getBoxSize() % 2 != 0:
            errors.append("Box size must be even!")

        if not self._inputHasAlign():
            errors.append("Input particles have no alignment!")

        if self._getBoxSize() < 128:
            errors.append("OPUS-DSD requires a box size > 128 x 128 pixels.")

        return errors

    # --------------------------- UTILS functions -----------------------------
    def _getAnalyzeArgs(self, epoch):
        return [
            self.getOutputDir(),
            '%d' % epoch,
            '%d' % self.numPCs.get(),
            '%d' % self.ksamples,
        ]
    
    def _getEvalVolArgs(self, epoch, i=None, sampleMode=KMEANS):
        if self.outApix == -1:
            self.samplingRate = params.Float(self.inputParticles.get().getSamplingRate())
        else:
            self.samplingRate = params.Float(self.outApix)

        if sampleMode == KMEANS:
            return [
                self.getOutputDir(),
                '%d' % epoch,
                '%d' % self.ksamples,
                '%4.2f' % self.samplingRate,
                'kmeans'
            ]
        else:
            return [
                self.getOutputDir(),
                '%d' % epoch,
                '%d' % i,
                '%4.2f' % self.samplingRate,
                'pc'
            ]

    def _runProgram(self, program, args, fromCryodrgn=True):
        gpus = ','.join(str(i) for i in self.getGpuList())
        self.runJob(Plugin.getProgram(program, gpus, 
                                      fromCryodrgn=fromCryodrgn), ' '.join(args))

    def _getVolumes(self, sampleMode=KMEANS, vols=None, zValues=None):
        """ Returns a list of volume names and their zValues. """
        if vols is None:
            vols = []

        if zValues is None:
            zValues = []

        if self.hasMultLatentVars():
            if sampleMode == KMEANS:
                fn = 'output_volN_km'
                num = self.ksamples.get()
                zValue = 'z_valuesN_km'
                zValues.extend(self._getVolumeZvalues(self._getFileName(zValue, ksamples=num)))
            else:
                fn = 'output_volN_pc'
                num = 10
                zValue = 'z_valuesN_pc'
                for i in range(1, self.numPCs.get()+1):
                    zValues.extend(self._getVolumeZvalues(self._getFileName(zValue, i=i)))
        else:
            fn = 'output_vol'
            num = 10
            zValue = 'z_values'
            zValues.extend(self._getVolumeZvalues(self._getFileName(zValue)))

        for volId in range(num):
            if self.hasMultLatentVars():
                if sampleMode == KMEANS:
                    volFn = self._getFileName(fn, ksamples=num, epoch=self._epoch,
                                              id=volId)
                    vols = self._appendVolumes(vols, volFn)
                else:
                    for i in range(1, self.numPCs.get()+1):
                        volFn = self._getFileName(fn, i=i, epoch=self._epoch,
                                                  id=volId)
                        vols = self._appendVolumes(vols, volFn)
            else:
                volFn = self._getFileName(fn, epoch=self._epoch, id=volId)
                vols = self._appendVolumes(vols, volFn)

        return vols, zValues

    def _getParticlesZvalues(self):
        """
        Read from z.pkl file the particles z_values
        :return: a numpy array with the particles z_values
        """
        zEpochFile = self.getEpochZFile(self._epoch)
        with open(zEpochFile, 'rb') as f:
            zValues = pickle.load(f)
        return zValues

    def _createParticleSet(self):
        """
        Create a set of particles with the associated z_values
        :return: a set of particles
        """
        inImgSet = self.inputParticles.get()
        zValues = iter(self._getParticlesZvalues())
        outImgSet = self._createSetOfParticles()
        outImgSet.copyInfo(inImgSet)
        outImgSet.copyItems(inImgSet, updateItemCallback=self._setZValues,
                            itemDataIterator=zValues)
        setattr(outImgSet, WEIGHTS, pwobj.String(self._getFileName('weights')))
        setattr(outImgSet, CONFIG, pwobj.String(self._getFileName('config')))

        return outImgSet

    def _setZValues(self, item, row=None):
        vector = pwobj.CsvList()
        # We assume that each row "i" of z_values corresponds to each
        # particle with ID "i"
        vector._convertValue(list(row))
        setattr(item, Z_VALUES, vector)

    def _getVolumeZvalues(self, zValueFile):
        """
        Read from z_values.txt file the volume z_values
        :return: a list with the volumes z_values
        """
        return np.loadtxt(zValueFile, dtype=float).tolist()

    def _createVolumeSet(self, files, zValues, path, samplingRate,
                         updateItemCallback=None):
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
            # We assume that each row "i" of z_values corresponds to each
            # volumes with ID "i"
            volZValues = zValues[volId]
            vector._convertValue(volZValues)
            # Creating a new column in the volumes with the z_value
            setattr(vol, Z_VALUES, vector)
            if updateItemCallback:
                updateItemCallback(vol)
            volSet.append(vol)
            volId += 1

        return volSet

    def getOutputDir(self, *paths):
        return os.path.join(self._getPath('output'), *paths)

    def getEpochZFile(self, epoch):
        return self.getOutputDir('z.%d.pkl' % epoch)

    def getLastEpoch(self):
        """ Return the last iteration number. """
        epoch = None
        self._epochRegex = re.compile(r'z.(\d).pkl')
        files = sorted(glob(self.getEpochZFile(0).replace('0', '*')))
        if files:
            f = files[-1]
            s = self._epochRegex.search(f)
            if s:
                epoch = int(s.group(1))  # group 1 is a digit iteration number

        return epoch

    def hasMultLatentVars(self):
        return self.zDim > 1

    def _inputHasAlign(self):
        return self._getInputParticles().hasAlignmentProj()

    def _appendVolumes(self, vols, volFn):
        if os.path.exists(volFn):
            vols.append(volFn)
        else:
            raise FileNotFoundError("Volume %s does not exists. \n"
                                    "Please select a valid epoch "
                                    "number." % volFn)
        return vols
