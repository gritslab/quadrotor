#! /usr/bin/env python
"""
@brief This is a testing script for the quad.py module.

@author Rowland O'Flaherty
@date 07/13/2014
@version: 0.1
@copyright: Copyright (C) 2014, see the LICENSE file
"""

import copy
import numpy as np
from numpy import pi as PI
from numpy import sin as sin
from numpy import cos as cos
from numpy import tan as tan
from numpy import sqrt

# import matplotlib
# from matplotlib import pyplot as plt
# matplotlib.use("TKAgg")
# from pylab import *

from control import lqr

import robotics.mechanics as mechanics
import robotics.rigid as rigid
import robotics.frames as frames
import utility.mmath as mmath
import quadrotor.quad as quad
import quadrotor.quad_model as quad_model
from quadrotor.sim_params import *

print ""
print "*==================================*"
print "|  TEST SCRIPT FOR quad.py MODULE  |"
print "*==================================*"
print ""

print ""
print "*---------------*"
print "|  Quad Class  |"
print "*--------------*"
print ""

print "# Constructors for the frames.Quad class:"
print ""

Q0 = quad.Quad()
print ">> Q0 = quad.Quad()"
print "Q0 = \n" + str(Q0)
print ""

Q1 = quad.Quad(params=PUFFIN_PARAMS,
               position=(0, 0, 0),
               attitude=(0, 0, 0),
               linear_vel=(0, 0, 0),
               attitude_vel=(0, 0, 0))
print ">> Q1 = quad.Quad(quad_params=PUFFIN_PARAMS,"
print ">>                position=(0, 0, 0),"
print ">>                attitude=(0, 0, 0),"
print ">>                linear_vel=(0, 0, 0),"
print ">>                attitude_vel=(0, 0, 0))"
print "Q1 = \n" + str(Q1)
print ""

print ">> Q1.position"
print str(Q1.position)
print

print ">> Q1.attitude"
print Q1.val2str(Q1.attitude)
print ""

print ">> Q1.linear_vel"
print str(Q1.linear_vel)
print ""

print ">> Q1.angular_vel"
print str(Q1.angular_vel)
print ""

print ">> Q1.attitude_vel"
print Q1.val2str(Q1.attitude_vel)
print ""

print ">> Q1.kinetic"
print str(Q1.kinetic)
print ""

print ">> Q1.potential"
print str(Q1.potential)
print ""

print ">> Q1.energy"
print str(Q1.energy)
print ""

print ">> Q1.lagragian"
print str(Q1.lagragian)
print ""

print ">> Q1.coriolis"
print str(Q1.coriolis)
print ""

Q1.pwm = (0, 0, 0, 0)
print ">> Q1.pwm = (0, 0, 0, 0)"
print "Q1.pwm = " + str(Q1.pwm)
print ""

Q1.force = -Q1.G * Q1.M
print ">> Q1.force = -Q1.G * Q1.M"
print ">> Q1.force = " + str(Q1.force)
print ""

print ">> Q1.torque"
print str(Q1.torque)
print ""

print ">> Q1.torque_attitude"
print str(Q1.torque_attitude)
print ""

print ">> Q1.input"
print str(Q1.input)
print ""

print ">> Q1.state"
print Q1.val2str(Q1.state)
print ""

print ">> Q1.state_dot"
print Q1.val2str(Q1.state_dot)
print ""

print ""
print "*-----------*"
print "|  Testing  |"
print "*-----------*"
print ""

Q = quad.Quad(params=PUFFIN_PARAMS, time=0,
              position=INIT_POSITION,
              attitude=INIT_ATTITUDE,
              linear_vel=INIT_LINEAR_VEL,
              attitude_vel=INIT_ATTITUDE_VEL,
              pwm=None)

self = Q

# (f_trans, f_rot, A_trans, B_trans, A_rot, B_rot) = quad_model.quad_dynamics(
#                                                         Q1.G,
#                                                         Q1.M,
#                                                         Q1.I[0, 0],
#                                                         Q1.I[1, 1],
#                                                         Q1.I[2, 2],
#                                                         Q1.lin_vel.x,
#                                                         Q1.lin_vel.y,
#                                                         Q1.lin_vel.z,
#                                                         Q1.roll,
#                                                         Q1.pitch,
#                                                         Q1.yaw,
#                                                         Q1.roll_rate,
#                                                         Q1.pitch_rate,
#                                                         Q1.yaw_rate,
#                                                         Q1.force.x,
#                                                         Q1.force.y,
#                                                         Q1.force.z,
#                                                         Q1.torque.x,
#                                                         Q1.torque.y,
#                                                         Q1.torque.z)

# (x_ddot, y_ddot, z_ddot, phi_ddot, theta_ddot, psi_ddot) = quad.quad_dynamics(
#                                                                 Q1.G,
#                                                                 Q1.M,
#                                                                 Q1.I,
#                                                                 Q1.roll,
#                                                                 Q1.pitch,
#                                                                 Q1.yaw,
#                                                                 Q1.roll_rate,
#                                                                 Q1.pitch_rate,
#                                                                 Q1.yaw_rate,
#                                                                 Q1.force.array,
#                                                                 Q1.torque.array)

# import random
# Q = quad.Quad(quad_params=QUAD_PARAMS, position=(0, 0, 10), lin_vel=(0, 0, 0), ang_vel=(0, 0, 0), reference=L)

# # Q.roll = PI/2*random.random() - PI/4
# # Q.pitch = PI/2*random.random() - PI/4
# # Q.yaw = 2*PI*random.random() - PI

# Q.roll = 1
# Q.pitch = -.5
# Q.yaw = 1

# print "Roll: " + str(Q.roll)
# print "Pitch: " + str(Q.pitch)
# print "Yaw: " + str(Q.yaw)

# # Q.refresh()
# # Q.update()
# self = Q

# if 1:
#     t = []
#     phi = []
#     theta = []
#     psi = []
#     pwm = []
#     x = []
#     y = []
#     z = []

#     tf = 2.5
#     while Q.t < tf:
#         Q.update()
#         t.append(Q.t)
#         phi.append(Q.roll)
#         theta.append(Q.pitch)
#         psi.append(Q.yaw)
#         pwm.append(Q.pwm)
#         x.append(Q.x)
#         y.append(Q.y)
#         z.append(Q.z)

#     figure(1)
#     subplot(311)
#     plot(t, phi, color="blue", linewidth=2, label='Roll')
#     plot(t, theta, color="green", linewidth=2, label='Pitch')
#     plot(t, psi, color="red", linewidth=2, label='Yaw')
#     xlim(0, tf)
#     legend()
#     subplot(312)
#     plot(t, x, color="blue", linewidth=2, label='x')
#     plot(t, y, color="green", linewidth=2, label='y')
#     plot(t, z, color="red", linewidth=2, label='z')
#     xlim(0, tf)
#     legend()
#     subplot(313)
#     line_handle = plot(t, pwm, linewidth=2)
#     xlim(0, tf)
#     legend(iter(line_handle), ('u1', 'u2', 'u3', 'u4'))
#     show()
