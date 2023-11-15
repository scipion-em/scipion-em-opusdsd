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
from enum import Enum

import pyworkflow.utils as pwutils
from pyworkflow.plugin import Domain
from pyworkflow.constants import PROD
import pyworkflow.protocol.params as params
from pwem.constants import ALIGN_PROJ, ALIGN_NONE
from pwem.protocols import ProtProcessParticles

from .. import Plugin
from ..objects import CryoDrgnParticles
from ..constants import OPUSDSD_HOME

convert = Domain.importFromPlugin('relion.convert', doRaise=True)


class outputs(Enum):
    outputCryoDrgnParticles = CryoDrgnParticles


class OpusDsdProtPreprocess(ProtProcessParticles):
    """ Protocol to downsample a particle stack and prepare alignment/CTF parameters.
    """
    _label = 'preprocess'
    _devStatus = PROD
    _possibleOutputs = outputs

    def _createFilenameTemplates(self):
        """ Centralize how files are called. """

        def out(*p):
            return os.path.join(self._getPath('output_particles'), *p)

        myDict = {
            'input_parts': self._getExtraPath('input_particles.star'),
            'output_folder': out(),
            'output_parts': out('particles.%d.mrcs' % self._getBoxSize()),
            'output_txt': out('particles.%d.ft.txt' % self._getBoxSize()),
            'output_poses': out('poses.pkl'),
            'output_ctfs': out('ctfs.pkl'),
        }

        self._updateFilenamesDict(myDict)

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addHidden('usePreprocess', params.BooleanParam, default=True)
        form.addSection(label='Input')
        form.addParam('inputParticles', params.PointerParam,
                      pointerClass='SetOfParticles',
                      pointerCondition='hasCTF',
                      label="Input particles", important=True,
                      help='Select a set of particles from a consensus C1 '
                           '3D refinement.')

        form.addParam('doScale', params.BooleanParam, default=True,
                      label='Downsample particles?')

        form.addParam('scaleSize', params.IntParam, default=128,
                      condition='doScale',
                      validators=[params.Positive],
                      label='New box size (px)',
                      help='New box size in pixels, must be even.')

        form.addParam('doWindow', params.BooleanParam, default=True,
                      expertLevel=params.LEVEL_ADVANCED,
                      label="Apply circular mask?")

        form.addParam('winSize', params.FloatParam, default=0.85,
                      expertLevel=params.LEVEL_ADVANCED,
                      condition='doWindow',
                      label="Window size",
                      help="Circular windowing mask inner radius")

        form.addParam('chunk', params.IntParam, default=0,
                      label='Split in chunks',
                      help='Chunk size (in # of images) to split '
                           'particle stack when saving.')

        form.addParam('doInvert', params.BooleanParam, default=True,
                      expertLevel=params.LEVEL_ADVANCED,
                      label="Are particles white?")

        form.addParallelSection(threads=16, mpi=0)

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._createFilenameTemplates()
        self._insertFunctionStep(self.convertInputStep)
        self._insertFunctionStep(self.runDownSampleStep)
        self._insertFunctionStep(self.runParseMdStep)
        self._insertFunctionStep(self.createOutputStep)

    # --------------------------- STEPS functions -----------------------------
    def convertInputStep(self):
        """ Create a star file as expected by cryoDRGN."""
        outputFolder = self._getFileName('output_folder')
        pwutils.cleanPath(outputFolder)
        pwutils.makePath(outputFolder)

        imgSet = self.inputParticles.get()
        # Create links to binary files and write the relion .star file
        alignType = ALIGN_PROJ if self._inputHasAlign() else ALIGN_NONE
        convert.writeSetOfParticles(
            imgSet, self._getFileName('input_parts'),
            outputDir=self._getExtraPath(), alignType=alignType)

    def runDownSampleStep(self):
        self._runProgram('preprocess', self._getPreprocessArgs())

    def runParseMdStep(self):
        if self._inputHasAlign():
            self._runProgram('parse_pose_star', self._getParsePosesArgs())
        self._runProgram('parse_ctf_star', self._getParseCtfArgs())

    def createOutputStep(self):
        poses = self._getFileName('output_poses') if self._inputHasAlign() else None
        output = CryoDrgnParticles(filename=self._getFileName('output_txt'),
                                   poses=poses,
                                   ctfs=self._getFileName('output_ctfs'),
                                   dim=self._getBoxSize() + 1,
                                   samplingRate=self._getSamplingRate())

        self._defineOutputs(**{outputs.outputCryoDrgnParticles.name: output})
        self._defineSourceRelation(self.inputParticles, output)

    # --------------------------- INFO functions ------------------------------
    def _summary(self):
        summary = []
        self._createFilenameTemplates()
        if not self.isFinished():
            summary.append("Output not ready")
        else:
            poses = "poses and" if self._inputHasAlign() else ""
            summary.append(f"Created {poses} ctf files for cryoDRGN.")

        return summary

    def _validate(self):
        errors = []

        particles = self._getInputParticles()

        if self.doScale and self.scaleSize > particles.getXDim():
            errors.append("You cannot upscale particles!")

        if self._getBoxSize() % 2 != 0:
            errors.append("Box size must be even!")

        return errors

    def _warnings(self):
        warnings = []

        if not self._inputHasAlign():
            warnings.append("Input particles have no alignment, you will only "
                            "be able to use the output for ab initio training!")

        if self._getBoxSize() % 8 != 0:
            warnings.append("CryoDRGN mixed-precision (AMP) training will "
                            "require box size divisible by 8. Alternatively, "
                            "you will have to provide --no-amp option.")

        return warnings

    # --------------------------- UTILS functions -----------------------------
    def _getPreprocessArgs(self):
        args = ['%s ' % os.path.abspath(self._getFileName('input_parts')),
                '-o %s ' % os.path.abspath(self._getFileName('output_parts')),
                '-D %d' % self._getBoxSize(),
                '--window-r %0.2f' % self.winSize if self.doWindow else '--no-window',
                '--max-threads %d ' % self.numberOfThreads
                ]

        if not self.doInvert:
            args.append('--uninvert-data')

        if self.chunk > 0:
            args.append('--chunk %d ' % self.chunk)

        return args

    def _getParsePosesArgs(self):
        args = ['%s ' % self._getFileName('input_parts'),
                '-o %s ' % self._getFileName('output_poses')]

        return args

    def _getParseCtfArgs(self):
        args = ['%s ' % self._getFileName('input_parts'),
                '-o %s ' % self._getFileName('output_ctfs'),
                '--ps 0']  # required due to cryodrgn parsing bug

        return args

    def _getInputParticles(self):
        return self.inputParticles.get()

    def _getBoxSize(self):
        if self.doScale:
            return self.scaleSize.get()
        else:
            return self._getInputParticles().getXDim()

    def _getSamplingRate(self):
        inputSet = self._getInputParticles()
        oldSampling = inputSet.getSamplingRate()
        scaleFactor = self._getScaleFactor(inputSet)

        return oldSampling * scaleFactor

    def _getScaleFactor(self, inputSet):
        return inputSet.getXDim() / self._getBoxSize()

    def _inputHasAlign(self):
        return self._getInputParticles().hasAlignmentProj()

    def _runProgram(self, program, args):
        self._enterDir(os.path.join(Plugin.getVar(OPUSDSD_HOME), "commands"))
        self.runJob(Plugin.getProgram(program), ' '.join(args))
