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

from pyworkflow.tests import BaseTest, DataSet, setupTestProject
from pyworkflow.utils import magentaStr
from pwem.protocols import (ProtImportParticles, ProtSubSet, ProtImportMask)
from xmipp3.legacy.tests.test_protocols_subtract_projection import samplingRate
from xmipp3.protocols import (XmippProtCropResizeParticles, XmippResizeHelper,
                              XmippProtCreateMask3D, XmippProtCropResizeVolumes)

from ..protocols import OpusDsdProtTrain, OpusDsdProtAnalyze
from ..constants import *

class TestOpusDsd(BaseTest):
    @classmethod
    def runImportParticlesStar(cls, parts, mag, samplingRate):
        """ Import particles from Relion star file. """
        print(magentaStr("\n==> Import particles from Relion star file:"))
        cls.protImportPart = cls.newProtocol(ProtImportParticles,
                                         importFrom=ProtImportParticles.IMPORT_FROM_RELION,
                                         starFile=parts,
                                         #sqliteFile=parts,
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
                                         nElements=10000)
        cls.launchProtocol(cls.protPartSubset)
        return cls.protPartSubset

    @classmethod
    def runResizeParticles(cls, parts):
        """ Resize particles from previous Import. """
        print(magentaStr("\n==> Resize particles from previous Import:"))
        cls.protResizePart = cls.newProtocol(XmippProtCropResizeParticles,
                                         inputParticles=parts, doResize=True,
                                         resizeOption=XmippResizeHelper.RESIZE_DIMENSIONS,
                                         resizeDim=64)
        cls.launchProtocol(cls.protResizePart)
        return cls.protResizePart

    @classmethod
    def runImportMask(cls, path, samplingRate):
        """ Import mask for selected particles. """
        print(magentaStr("\n==> Import mask for selected particles:"))
        cls.protImportMask = cls.newProtocol(ProtImportMask,
                                        maskPath=path,
                                        samplingRate=samplingRate)
        cls.launchProtocol(cls.protImportMask)
        return cls.protImportMask

    @classmethod
    def runResizeMask(cls, mask):
        """ Resize particles from previous Import. """
        print(magentaStr("\n==> Resize mask particles from previous Import:"))
        cls.protResizeMask = cls.newProtocol(XmippProtCropResizeVolumes,
                                         inputVolumes=mask, doResize=True,
                                         resizeOption=XmippResizeHelper.RESIZE_DIMENSIONS,
                                         resizeDim=64)
        cls.launchProtocol(cls.protResizeMask)
        return cls.protResizeMask

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        #cls.dataset = DataSet.getDataSet('FlexHub_Tutorials')
        cls.dataset = '/home/egarcia/Escritorio/for/scipion/data/tests/FlexHub_Tutorials'
        cls.partFn = cls.dataset + '/Advanced_Guide/particles_026609.star'
        cls.mask = cls.dataset + '/Advanced_Guide/reference_mask.mrc'
        cls.protImportPart = cls.runImportParticlesStar(cls.partFn, 50000, samplingRate=samplingRate)
        cls.protPartSubset = cls.runCreateParticlesSubset(cls.protImportPart.outputParticles)
        cls.protResizePart = cls.runResizeParticles(cls.protPartSubset.outputParticles)
        cls.protImportMask = cls.runImportMask(cls.mask, samplingRate=samplingRate)
        cls.protResizeMask = cls.runResizeMask(cls.protImportMask.outputMask)

    def testTrainingAnalysis(self):
        print(magentaStr("\n==> Testing OPUS-DSD - Training Ab-Initio:"))
        protTrain = self.newProtocol(OpusDsdProtTrain,
                                     useMask=True,
                                     abInitio=True,
                                     numEpochs=20,
                                     zDim=2)
        protTrain.inputParticles.set(self.protPartSubset.outputParticles)
        protTrain.inputMask.set(self.protImportMask.outputMask)
        self.launchProtocol(protTrain)

        print(magentaStr("\n==> Testing OPUS-DSD - Analysis:"))
        protAnalysis = self.newProtocol(OpusDsdProtAnalyze,
                                        zDim=12,
                                        sampleMode=PCA,
                                        PC=4)

        protAnalysis.inputParticles.set(self.protPartSubset.outputParticles)
        protAnalysis.opusDSDTrainingProtocol.set(protTrain)
        self.launchProtocol(protAnalysis)

        print(magentaStr("\n==> Testing OPUS-DSD - Training:"))
        protTrain2 = self.newProtocol(OpusDsdProtTrain,
                                     abInitio=False,
                                     numEpochs=20,
                                     zDim=12)
        protTrain2.inputParticles.set(self.protPartSubset.outputParticles)
        protTrain2.inputMask.set(self.protImportMask.outputMask)
        protTrain2.opusDSDTrainingProtocol.set(protTrain)
        self.launchProtocol(protTrain2)