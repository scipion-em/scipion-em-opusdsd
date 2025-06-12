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

import os

from pyworkflow.tests import BaseTest, DataSet, setupTestProject
from pyworkflow.utils import magentaStr
from pwem.protocols import (ProtImportParticles, ProtImportVolumes,
                            ProtSubSet, ProtUserSubSet)
from xmipp3.legacy.tests.test_protocols_subtract_projection import samplingRate
from xmipp3.protocols import (XmippProtCropResizeParticles, XmippResizeHelper,
                              XmippProtCropResizeVolumes, XmippProtCreateMask3D)

from ..protocols import OpusDsdProtTrain, OpusDsdProtAnalyze
from ..constants import *

class TestOpusDsd(BaseTest):
    @classmethod
    def runImportParticlesStar(cls, partStar, mag, samplingRate):
        """ Import particles from Relion star file. """
        print(magentaStr("\n==> Import particles from Relion star file:"))
        cls.protImportPart = cls.newProtocol(ProtImportParticles,
                                         importFrom=ProtImportParticles.IMPORT_FROM_SCIPION,
                                         sqliteFile=partStar,
                                         magnification=mag,
                                         samplingRate=samplingRate,
                                         haveDataBeenPhaseFlipped=False)
        cls.launchProtocol(cls.protImportPart)
        return cls.protImportPart

    @classmethod
    def runCreateParticlesSubset(cls, parts):
        """ Creation of a subset of particles from previous protocol. """
        print(magentaStr("\n==> Creation of a subset of particles from previous protocol:"))
        cls.protPartSubset = cls.newProtocol(ProtSubSet,
                                         inputFullSet=parts,
                                         chooseAtRandom=True,
                                         nElements=500)
        cls.launchProtocol(cls.protPartSubset)
        return cls.protPartSubset

    @classmethod
    def runResizeParticles(cls, parts):
        """ Resize particles from previous Import. """
        print(magentaStr("\n==> Resize particles from previous Import:"))
        cls.protResizePart = cls.newProtocol(XmippProtCropResizeParticles,
                                         inputParticles=parts, doResize=True,
                                         resizeOption=XmippResizeHelper.RESIZE_DIMENSIONS,
                                         resizeDim=128)
        cls.launchProtocol(cls.protResizePart)
        return cls.protResizePart

    @classmethod
    def runImportVolume(cls, path, samplingRate):
        """ Import volumes from Relion star file. """
        print(magentaStr("\n==> Import volume from Relion star file:"))
        cls.protImportVol = cls.newProtocol(ProtImportVolumes,
                                        filesPath=path,
                                        samplingRate=samplingRate)
        cls.launchProtocol(cls.protImportVol)
        return cls.protImportVol

    @classmethod
    def runResizeVolume(cls, vol):
        """ Resize volumes from previous Import. """
        print(magentaStr("\n==> Resize volume from previous Import:"))
        cls.protResizeVol = cls.newProtocol(XmippProtCropResizeVolumes,
                                        inputVolumes=vol, doResize=True,
                                        resizeOption=XmippResizeHelper.RESIZE_DIMENSIONS,
                                        resizeDim=128)
        cls.launchProtocol(cls.protResizeVol)
        return cls.protResizeVol

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        #cls.dataset = DataSet.getDataSet('FlexHub_Tutorials')
        cls.dataset = '/home/egarcia/Escritorio/for/scipion/data/tests/FlexHub_Tutorials'
        cls.partFn = cls.dataset + '/Starter_Guide/particles.sqlite'
        cls.vol = cls.dataset + '/Starter_Guide/AK.vol'
        cls.protImportPart = cls.runImportParticlesStar(cls.partFn, 50000, samplingRate=samplingRate)
        cls.protPartSubset = cls.runCreateParticlesSubset(cls.protImportPart.outputParticles)
        cls.protResizePart = cls.runResizeParticles(cls.protPartSubset.outputParticles)
        cls.protImportVol = cls.runImportVolume(cls.vol, samplingRate=samplingRate)
        cls.protResizeVol = cls.runResizeVolume(cls.protImportVol.outputVolume)

    def testTrainingAnalysis(self):
        print(magentaStr("\n==> Testing OPUS-DSD - Training Ab-Initio:"))
        protTrain = self.newProtocol(OpusDsdProtTrain,
                                     abInitio=PREPROCESS,
                                     numEpochs=50,
                                     zDim=4)
        protTrain.inputParticles.set(self.protResizePart.outputParticles)
        protTrain.inputVolume.set(self.protResizeVol.outputVol)
        self.launchProtocol(protTrain)

        print(magentaStr("\n==> Testing OPUS-DSD - Analysis:"))
        protAnalysis = self.newProtocol(OpusDsdProtAnalyze,
                                        zDim=4,
                                        viewEpoch=EPOCH_PENULTIMATE,
                                        numEpochs=50,
                                        sampleMode=PCS,
                                        PC=3)
        protAnalysis.inputParticles.set(self.protResizePart.outputParticles)
        protAnalysis.opusDSDTrainingProtocol.set(protTrain)
        self.launchProtocol(protAnalysis)

        print(magentaStr("\n==> Testing OPUS-DSD - Training:"))
        protTrain = self.newProtocol(OpusDsdProtTrain,
                                     abInitio=ANALYSIS,
                                     numEpochs=50,
                                     zDim=10)
        protTrain.inputParticles.set(self.protResizePart.outputParticles)
        protTrain.inputVolume.set(self.protResizeVol.outputVol)
        protTrain.opusDSDAnalysisProtocol.set(protAnalysis)
        self.launchProtocol(protTrain)