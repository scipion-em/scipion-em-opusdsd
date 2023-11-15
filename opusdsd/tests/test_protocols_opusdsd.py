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

from pyworkflow.tests import BaseTest, DataSet, setupTestProject
from pyworkflow.utils import magentaStr
from pwem.protocols import ProtImportParticles

from ..protocols import OpusDsdProtPreprocess, OpusDsdProtTrain


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
        cls.assertIsNotNone(protImport.outputParticles,
                            "SetOfParticles has not been produced.")

        return protImport

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dataset = DataSet.getDataSet('relion_tutorial')
        cls.partFn = cls.dataset.getFile('import/refine3d_case2/relion_data.star')
        cls.protImport = cls.runImportParticlesStar(cls.partFn, 50000, 3.54)

    def runPreprocess(self, protLabel, particles, **kwargs):
        print(magentaStr(f"\n==> Testing OPUS-DSD - {protLabel}:"))
        protPreprocess = self.newProtocol(OpusDsdProtPreprocess,
                                          objLabel=protLabel, **kwargs)
        protPreprocess.inputParticles.set(particles)
        return self.launchProtocol(protPreprocess)

    def checkPreprocessOutput(self, preprocessProt):
        """ Do some check on the output of the preprocess. """
        output = getattr(preprocessProt, 'outputOpusDsdParticles', None)
        self.assertIsNotNone(output)

        filename = output.filename.get()
        poses = output.poses.get()
        ctfs = output.ctfs.get()

        for f in [filename, poses, ctfs]:
            fn = os.path.join(self.proj.path, f)
            self.assertTrue(os.path.exists(fn))

        self.assertTrue(filename.endswith('.txt'))

    def checkTrainingOutput(self, trainingProt):
        output = getattr(trainingProt, 'Particles', None)
        self.assertIsNotNone(output)

        output2 = getattr(trainingProt, 'Volumes', None)
        self.assertIsNotNone(output2)

    def testPreprocess(self):
        parts = self.protImport.outputParticles

        preprocess1 = self.runPreprocess("preprocess scale=64", parts, scaleSize=64)
        self.checkPreprocessOutput(preprocess1)

        preprocess2 = self.runPreprocess("preprocess scale=50 with chunks", parts,
                                         scaleSize=50, chunk=200)
        self.checkPreprocessOutput(preprocess2)

    def testTraining(self):
        parts = self.protImport.outputParticles
        preprocess = self.runPreprocess("preprocess scale=64", parts, scaleSize=64)

        print(magentaStr("\n==> Testing OPUS-DSD - training:"))
        protTrain = self.newProtocol(OpusDsdProtTrain, numEpochs=3, zDim=2)
        protTrain.inputParticles.set(preprocess.outputOpusDsdParticles)
        self.launchProtocol(protTrain)
        self.checkTrainingOutput(protTrain)
