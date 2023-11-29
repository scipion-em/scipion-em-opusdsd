# **************************************************************************
# *
# * Authors:     Grigory Sharov (gsharov@mrc-lmb.cam.ac.uk) [1]
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

#from relion import ProtRelionCreateMask3D
from pyworkflow.tests import BaseTest, DataSet, setupTestProject
from pyworkflow.utils import magentaStr
from pwem.protocols import ProtImportParticles, ProtImportVolumes
from xmipp3.protocols import (XmippProtCropResizeParticles, XmippResizeHelper,
                              XmippProtCropResizeVolumes, XmippProtCreateMask3D)

from ..protocols import OpusDsdProtTrain


class TestOpusDsd(BaseTest):
    @classmethod
    def runImportParticlesStar(cls, partStar, mag, samplingRate):
        """ Import particles from Relion star file. """
        print(magentaStr("\n==> Importing data - particles from star:"))
        protImport = cls.newProtocol(ProtImportParticles,
                                     importFrom=ProtImportParticles.IMPORT_FROM_RELION,
                                     starFile=partStar,
                                     magnification=mag,
                                     samplingRate=samplingRate,
                                     haveDataBeenPhaseFlipped=False)
        cls.launchProtocol(protImport)
        return protImport
    
    @classmethod
    def runResizeParticles(cls, parts):
        """ Import particles from Relion star file. """
        print(magentaStr("\n==> Importing data - particles from star:"))
        protResize = cls.newProtocol(XmippProtCropResizeParticles,
                                     inputParticles=parts, doResize=True,
                                     resizeOption=XmippResizeHelper.RESIZE_DIMENSIONS,
                                     resizeDim=64)
        cls.launchProtocol(protResize)
        return protResize

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dataset = DataSet.getDataSet('relion_tutorial')
        cls.partFn = cls.dataset.getFile('import/refine3d_case2/relion_data.star')
        cls.protImport = cls.runImportParticlesStar(cls.partFn, 50000, 3.54)
        cls.protResize = cls.runResizeParticles(cls.protImport.outputParticles)

    def checkTrainingOutput(self, trainingProt):
        output = getattr(trainingProt, 'Particles', None)
        self.assertIsNotNone(output)

        output2 = getattr(trainingProt, 'Volumes', None)
        self.assertIsNotNone(output2)

    def testTraining(self):
        print(magentaStr("\n==> Testing OPUS-DSD - training:"))
        protTrain = self.newProtocol(OpusDsdProtTrain, numEpochs=3, zDim=12,
                                     downFrac=1.)
        protTrain.inputParticles.set(self.protResize.outputParticles)
        self.launchProtocol(protTrain)
        #self.checkTrainingOutput(protTrain)

    @classmethod
    def runCreateMask(cls, vol):
        cls.maskProt = cls.newProtocol(XmippProtCreateMask3D)
        cls.maskProt.inputVolume.set(vol)
        print(magentaStr("\n==> Testing xmipp - create mask 3d:"))
        cls.launchProtocol(cls.maskProt)
