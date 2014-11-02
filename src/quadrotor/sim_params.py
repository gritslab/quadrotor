"""
@brief Classes and functions for quadrotor parameters

@author Rowland O'Flaherty
@date 10/19/2014
@version: 0.1
@copyright: Copyright (C) 2014, see the LICENSE file
"""

import numpy as np
from numpy import pi as PI

from quadrotor.quad import QuadParams

# ==============================================================================
# Simulator Constants
# ==============================================================================

SAMPLE_TIME = .05

PUFFIN_PARAMS = QuadParams(name='PUFFIN',
                             M=1,
                             I=np.array([[0.01, 0, 0],
                                        [0, 0.01, 0],
                                        [0, 0, 0.01]]),
                             L=.330,
                             C_F=np.array(4*[.0007]),
                             C_T=np.array(4*[.00003]))

INIT_POSITION = (0, 0, 0)
INIT_ATTITUDE = (0, 0, 0)
INIT_LINEAR_VEL = (0, 0, 0)
INIT_ATTITUDE_VEL = (0, 0, 0)
