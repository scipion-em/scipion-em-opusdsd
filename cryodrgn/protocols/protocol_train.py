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

import os
import pickle
import numpy as np
import re
from glob import glob

import pyworkflow.utils as pwutils
from pyworkflow.constants import PROD
import pyworkflow.protocol.params as params
import pyworkflow.object as pwobj
from pwem.protocols import ProtProcessParticles
import pwem.objects as emobj

from cryodrgn import Plugin
from cryodrgn.constants import *


class CryoDrgnProtTrain(ProtProcessParticles):
    """
    Protocol to train cryoDRGN neural network.
    """
    _label = 'training VAE'
    _devStatus = PROD

    def _createFilenameTemplates(self):
        """ Centralize how files are called within the protocol. """

        def out(*p):
            return os.path.join(self.getOutputDir(f'analyze.{self._epoch}'), *p)

        myDict = {
            'output_vol': out('vol_%(id)03d.mrc'),
            'output_volN': out('kmeans%(ksamples)d/vol_%(id)03d.mrc'),
            'z_values': out('z_values.txt'),
            'z_valuesN': out('kmeans%(ksamples)d/z_values.txt'),
            'weights': self.getOutputDir(f'weights.{self._epoch}.pkl'),
            'config': self.getOutputDir('config.pkl')
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

        form.addParam('inputParticles', params.PointerParam,
                      pointerClass="CryoDrgnParticles",
                      label='CryoDrgn particles')

        form.addParam('zDim', params.IntParam, default=8,
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

        form.addSection(label='Analysis')
        form.addParam('viewEpoch', params.EnumParam,
                      choices=['last', 'selection'], default=EPOCH_LAST,
                      display=params.EnumParam.DISPLAY_LIST,
                      label="Epoch to analyze")

        form.addParam('epochNum', params.IntParam,
                      condition='viewEpoch==%d' % EPOCH_SELECTION,
                      label="Epoch number")

        form.addParam('ksamples', params.IntParam, default=20,
                      label='Number of K-means samples to generate',
                      help="*cryodrgn analyze* uses the k-means clustering "
                           "algorithm to partition the latent space into "
                           "regions (by default k=20 regions), and generate a "
                           "density map from the center of each of these "
                           "regions. The goal is to provide a tractable number "
                           "of representative density maps to visually inspect. ")

        form.addParallelSection(threads=16, mpi=0)

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.runTrainingStep)
        if self.viewEpoch == EPOCH_LAST:
            self._epoch = self.numEpochs.get() - 1
        else:
            self._epoch = self.epochNum.get()
        self._createFilenameTemplates()
        self._insertFunctionStep(self.runAnalysisStep, self._epoch)
        self._insertFunctionStep(self.createOutputStep)

    # --------------------------- STEPS functions -----------------------------
    def runTrainingStep(self):
        # Create output folder
        pwutils.cleanPath(self.getOutputDir())
        pwutils.makePath(self.getOutputDir())

        # Call cryoDRGN with the appropriate parameters
        self._runProgram('train_vae', self._getTrainingArgs())

    def runAnalysisStep(self, epoch):
        """ Run analysis step.
        Args:
            epoch: epoch number to be analyzed.
        """
        self._runProgram('analyze', self._getAnalyzeArgs(epoch))

    def createOutputStep(self):
        """ Create the protocol outputs. """
        # Creating a set of particles with z_values
        outImgSet = self._createParticleSet()
        self._defineOutputs(Particles=outImgSet)

        # Creating a set of volumes with z_values
        fn = self._getExtraPath('volumes.sqlite')
        samplingRate = self.inputParticles.get().getSamplingRate()
        files, zValues = self._getVolumes()
        setOfVolumes = self._createVolumeSet(files, zValues, samplingRate)
        self._defineOutputs(Volumes=setOfVolumes)
        self._defineSourceRelation(self.inputParticles.get(), setOfVolumes)

    # --------------------------- INFO functions ------------------------------
    def _summary(self):
        summary = ["Training VAE for %d epochs." % self.numEpochs]

        return summary

    def _validate(self):
        errors = []

        if self.viewEpoch == EPOCH_SELECTION:
            ep = self.epochNum.get()
            total = self.numEpochs.get()
            if ep > total:
                errors.append(f"You can analyse only epochs 1-{total}")

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

    def _getAnalyzeArgs(self, epoch):
        return [
            self.getOutputDir(),
            f"{epoch}",
            '--Apix %0.3f' % self.inputParticles.get().getSamplingRate(),
            '--ksample %d' % self.ksamples,
        ]

    def _runProgram(self, program, args):
        gpus = ','.join(str(i) for i in self.getGpuList())
        self.runJob(Plugin.getProgram(program, gpus), ' '.join(args))

    def _getVolumes(self):
        vols = []
        if self.hasMultLatentVars():
            fn = 'output_volN'
            num = self.ksamples.get()
            zValue = 'z_valuesN'
            zValues = self._getVolumeZvalues(self._getFileName(zValue,
                                                               ksamples=num))
        else:
            fn = 'output_vol'
            num = 10
            zValue = 'z_values'
            zValues = self._getVolumeZvalues(self._getFileName(zValue))

        for volId in range(num):
            if self.hasMultLatentVars():
                volFn = self._getFileName(fn, ksamples=num, epoch=self._epoch,
                                          id=volId)
            else:
                volFn = self._getFileName(fn, epoch=self._epoch, id=volId)

            if os.path.exists(volFn):
                vols.append(volFn)
            else:
                raise FileNotFoundError("Volume %s does not exists. \n"
                                        "Please select a valid epoch "
                                        "number." % volFn)

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
        cryoDRGParticles = self.inputParticles.get()
        ImgSet = self.getProject().getProtocol(cryoDRGParticles.getObjParentId()).inputParticles.get()
        outImgSet = self._createSetOfParticles()
        outImgSet.copyInfo(ImgSet)
        outImgSet.copyItems(ImgSet, updateItemCallback=self._setZValues)

        setattr(outImgSet, WEIGHTS, pwobj.String(self._getFileName('weights')))
        setattr(outImgSet, CONFIG, pwobj.String(self._getFileName('config')))

        return outImgSet

    def _setZValues(self, item, row=None):
        zValues = self._getParticlesZvalues()
        vector = pwobj.CsvList()
        # We assume that each row "i" of z_values corresponds to each
        # particle with ID "i"
        vector._convertValue(list(zValues[item.getObjId()-1]))
        setattr(item, Z_VALUES, vector)

    def _getVolumeZvalues(self, zValueFile):
        """
        Read from z_values.txt file the volume z_values
        :return: a list with the volumes z_values
        """
        return np.loadtxt(zValueFile, dtype=str).tolist()

    def _createVolumeSet(self, files, zValues, samplingRate,
                         updateItemCallback=None):
        """
        Create a set of volume with the associated z_values
        :param files: list of the volumes path
        :param zValues: list with the volumes z_values
        :param samplingRate: volumes sampling rate
        :return: a set of volumes
        """
        volSet = self._createSetOfVolumes()
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
            vector._convertValue(zValues[volId])
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
        self._epochRegex = re.compile('z.(\d).pkl')
        files = sorted(glob(self.getEpochZFile(0).replace('0', '*')))
        if files:
            f = files[-1]
            s = self._epochRegex.search(f)
            if s:
                epoch = int(s.group(1))  # group 1 is a digit iteration number

        return epoch

    def hasMultLatentVars(self):
        return self.zDim > 1
