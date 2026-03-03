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

from pyworkflow.tests import BaseTest, setupTestProject
from pyworkflow.utils import magentaStr
from pyworkflow import Config
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
                                         nElements=1000)
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
                                         resizeDim=128)
        cls.launchProtocol(cls.protResizeMask)
        return cls.protResizeMask

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dataset = os.path.join(os.getcwd(), os.path.join(Config.SCIPION_HOME, 'data/tests/FlexHub_Tutorials'))
        cls.partFn = os.path.join(cls.dataset, 'Advanced_Guide/particles_026609.star')
        cls.mask = os.path.join(cls.dataset, 'Advanced_Guide/reference_mask.mrc')
        cls.protImportPart = cls.runImportParticlesStar(cls.partFn, 50000, samplingRate=samplingRate)
        cls.protPartSubset = cls.runCreateParticlesSubset(cls.protImportPart.outputParticles)
        cls.protResizePart = cls.runResizeParticles(cls.protPartSubset.outputParticles)
        cls.protImportMask = cls.runImportMask(cls.mask, samplingRate=samplingRate)
        cls.protResizeMask = cls.runResizeMask(cls.protImportMask.outputMask)

    def testTrainingAnalysis(self):
        print(magentaStr("\n==> Testing OPUS-DSD - Initial Training:"))
        self.protTrain = self.newProtocol(OpusDsdProtTrain, abInitio=True, numEpochs=10, zDim=8, templateres=80)
        self.protTrain.inputParticles.set(self.protResizePart.outputParticles)
        self.protTrain.inputMask.set(self.protResizeMask.outputVol)
        self.launchProtocol(self.protTrain)

        print(magentaStr("\n==> Testing OPUS-DSD - Analysis (KMEANS):"))
        self.protAnalysis = self.newProtocol(OpusDsdProtAnalyze, sampleMode=KMEANS, ksamples=24)
        self.protAnalysis.inputParticles.set(self.protResizePart.outputParticles)
        self.protAnalysis.opusDSDTrainingProtocol.set(self.protTrain)
        self.launchProtocol(self.protAnalysis)

        print(magentaStr("\n==> Testing OPUS-DSD - Analysis (PCA):"))
        self.protAnalysis2 = self.newProtocol(OpusDsdProtAnalyze, sampleMode=PCA, psamples=24)
        self.protAnalysis2.inputParticles.set(self.protResizePart.outputParticles)
        self.protAnalysis2.opusDSDTrainingProtocol.set(self.protTrain)
        self.launchProtocol(self.protAnalysis2)