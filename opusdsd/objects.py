# **************************************************************************
# *
# * Authors:     J.M. de la Rosa Trevin (delarosatrevin@scilifelab.se) [1]
# *              Grigory Sharov (gsharov@mrc-lmb.cam.ac.uk) [2]
# *
# * [1] SciLifeLab, Stockholm University
# * [2] MRC Laboratory of Molecular Biology, MRC-LMB
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

import pyworkflow.object as pwobj
from pwem.objects import EMObject


class CryoDrgnParticles(EMObject):

    def __init__(self, filename=None, poses=None, ctfs=None,
                 dim=None, samplingRate=None, **kwargs):
        EMObject.__init__(self, **kwargs)

        self.filename = pwobj.String(filename)
        self.poses = pwobj.String(poses)
        self.ctfs = pwobj.String(ctfs)
        self.samplingRate = pwobj.Float(samplingRate)
        self.dim = pwobj.Integer(dim)

    def __str__(self):
        return ('CryoDrgnParticles (%d x %d, %0.2f Å/px)'
                % (self.dim, self.dim, self.samplingRate))

    def getSamplingRate(self):
        return self.samplingRate.get()

    def getXDim(self):
        return self.dim.get()
