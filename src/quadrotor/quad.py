"""
@brief Classes and functions for quadrotor

@author Rowland O'Flaherty
@date 08/13/2014
@version: 0.1
@copyright: Copyright (C) 2014, see the LICENSE file
"""

import numpy as np
from numpy import pi as PI
from numpy import sin
from numpy import cos
from numpy import tan
from numpy import sqrt

# Documentation: http://python-control.sourceforge.net/manual-0.5a/matlab_strings.html#statefbk.lqr
# Download: http://sourceforge.net/p/python-control/wiki/Download/
#           https://github.com/repagh/Slycot
from control import lqr

import utility.mmath as mmath
import robotics.mechanics as mechanics
import robotics.frames as frames
import robotics.rigid as rigid
import quadrotor.quad_model as quad_model

class QuadParams(object):
    """
    Quadrotor paramerters

    @type g: float
    @param g: Gravitational acceleration constant (m/s^2) without direction

    @type m: float
    @param m: Mass (kg)

    @type cm: numpy.array (3 x 1)
    @param cm: Center of mass (m)

    @type I: numpy.array (3 x 3)
    @param I: Moment of inertia (kg m^2)
    """

    # Properties
    _G = 9.80665  # Gravity acceleration [m/s^2]
    _M = 1.  # Mass [kg]
    _I = np.identity(3)  # Moment of inertia [kg m^2]
    _L = 1.  # Arm [m]
    _C_F = np.ones(4)  # Constant from motor PWM^2 value to motor force
    _C_T = np.ones(4)  # Constant from motor PWM^2 value to motor torque

    # Constructor
    # def __init__(self, name=None):
    #     self.name = name

    def __init__(self, name=None, M=1, I=np.identity(3), L=1,
                 C_F=np.ones(4), C_T=np.ones(4)):
        self.name = name
        self.M = M
        self.I = I
        self.L = L
        self.C_F = C_F
        self.C_T = C_T

    # Property getters and setters

    # Mass
    @property
    def G(self):  # pylint: disable=C0111
        return self._G

    @property
    def M(self):  # pylint: disable=C0111
        return self._M
    @M.setter
    def M(self, value):  # pylint: disable=C0111
        if value > 0:
            self._M = float(value)
        else:
            raise Exception("Invalid argument:\n" + str(value) + \
                    "\n" + "Valid types: float > 0")

    # Moment of inertia
    @property
    def I(self):  # pylint: disable=C0111
        return self._I
    @I.setter
    def I(self, value):  # pylint: disable=C0111
        if isinstance(value, np.ndarray) and value.shape == (3, 3):
            self._I = value
        else:
            raise Exception("Invalid argument:\n" + str(value) + \
                    "\n" + "Valid types: 3x3 numpy.array")

    @property
    def L(self):  # pylint: disable=C0111
        return self._L
    @L.setter
    def L(self, value):  # pylint: disable=C0111
        if value > 0:
            self._L = float(value)
        else:
            raise Exception("Invalid argument:\n" + str(value) + \
                    "\n" + "Valid types: float > 0")

    @property
    def C_F(self):  # pylint: disable=C0111
        return self._C_F
    @C_F.setter
    def C_F(self, value):  # pylint: disable=C0111
        if len(value) == 4:
            self._C_F = value
        else:
            raise Exception("Invalid argument:\n" + str(value) + \
                    "\n" + "Valid types: 1x4 numpy.ndarray")

    @property
    def C_T(self):  # pylint: disable=C0111
        return self._C_T
    @C_T.setter
    def C_T(self, value):  # pylint: disable=C0111
        if len(value) == 4:
            self._C_T = value
        else:
            raise Exception("Invalid argument:\n" + str(value) + \
                    "\n" + "Valid types: 1x4 numpy.ndarray")


class Quad(QuadParams, frames.AircraftState):
    """
    Quadrotor class
    """

    # Properties
    _pwm = None
    _init_state = np.zeros((12, 1))
    _init_pwm = (0, 0, 0, 0)
    _state_desired = np.zeros((12, 1))
    _input_desired = np.zeros((4,1))
    _state_err_cum_sum = np.zeros((12, 1))

    _state_desired_size = np.array([[.2, .2, .2,
                                     .07, .07, .07,
                                     .2, .2, .2,
                                     .1, .1, .1]])

    _max_lin_vel = 100
    _max_ang_vel = 20

    _finite_state = []
    _finite_state_set = [['LANDED',
                          'HOLDING',
                          'MOVING']]
    _finite_state_dim = 1

    # Constructor
    def __init__(self, params=QuadParams(), time=0,
                 position=(0, 0, 0), attitude=(0, 0, 0),
                 linear_vel=(0, 0, 0), attitude_vel=(0, 0, 0),
                 pwm=None, reference=frames.LOCAL):

        QuadParams.__init__(self, name=params.name,
                            M=params.M,
                            I=params.I,
                            L=params.L,
                            C_F=params.C_F,
                            C_T=params.C_T)

        frames.AircraftState.__init__(self, name=params.name, time=time,
                                      position=position,
                                      attitude=attitude,
                                      linear_vel=linear_vel,
                                      attitude_vel=attitude_vel,
                                      reference=reference)


        self._finite_state = ['LANDED']
        self._init_state = self.state
        self.state_desired = self.state

        if pwm is None:
            self.pwm = (0, 0, 0, 0)
            self.force = rigid.Vector(z=-self.G * self.M, reference=self)
        else:
            self.pwm = pwm
        self._init_pwm = self.pwm

        # # LQR attitude controller
        # self.controller = self.lqr_attitude_control

        # LQR position and attitude controller
        self.controller = self.lqr_position_attitude_control

        # # CLF attitude controller
        # self.controller = self.clf_attitude_control
        # self.state_desired[3, 0] = 1
        # self.state_desired[4, 0] = 0
        # self.state_desired[5, 0] = 0

        # # Diffeomorphism full state controller
        # self.controller = self.diffeomorphism_full_state_control

        # # Altitude controller
        # self.controller = self.pid_control

        # self.pid_P = np.zeros((self.input_dim, self.state_dim/2))
        # self.pid_I = np.zeros((self.input_dim, self.state_dim/2))
        # self.pid_D = np.zeros((self.input_dim, self.state_dim/2))

        # self.pid_P[2, 2] = .5
        # self.pid_I[2, 2] = .01
        # self.pid_D[2, 2] = .5

        # self._state_desired[2] = .5
        # self._input_desired = self.input


    # Property getters and setters
    @property
    def finite_state_set(self):
        return self._finite_state_set

    @property
    def finite_state(self):
        return self._finite_state
    @finite_state.setter
    def finite_state(self, value):
        for index in range(0, self._finite_state_dim):
            if value[index] in self._finite_state_set[index]:
                self._finite_state[index] = value[index]
            else:
                raise Exception("Invalid state: " + str(value[index]))

    @property
    def state_dim(self):
        return 12

    @property
    def input_dim(self):
        return 4

    @property
    def controller(self):
        return self._controller
    @controller.setter
    def controller(self, value):
        if hasattr(value, '__call__'):
            self._controller = value
        else:
            raise Exception("Invalid argument: " + str(value) +  "\n" + \
                            "Valid types: function")


    @property
    def state_desired(self):
        return self._state_desired
    @state_desired.setter
    def state_desired(self, value):
        if np.shape(value) == (12, 1):
            self._state_desired = value
            self._state_err_cum_sum = np.zeros((12, 1))
        else:
            raise Exception("Invalid argument: " + str(value) +  "\n" + \
                            "Valid types: 12x1 numpy.array")

    @property
    def at_state_desired(self):
        return np.all(abs(self.state_err) < self._state_desired_size)


    @property
    def init_state(self):
        return self._init_state

    @property
    def input_desired(self):
        return self._input_desired
    @input_desired.setter
    def input_desired(self, value):
        if np.shape(value) == (4, 1):
            self._input_desired = value
        else:
            raise Exception("Invalid argument: " + str(value) +  "\n" + \
                            "Valid types: 4x1 numpy.array")

    @property
    def state_err(self):
        return (self.state_desired - self.state)

    @property
    def state_err_cum_sum(self):
        return self._state_err_cum_sum

    @property
    def init_pwm(self):
        return self._init_pwm

    @property
    def pwm(self):  # pylint: disable=C0111
        return self._pwm
    @pwm.setter
    def pwm(self, value):  # pylint: disable=C0111
        if len(value) == 4:
            self._pwm = value
        else:
            raise Exception("Invalid argument:\n" + str(value) + \
                    "\n" + "Valid types: tuple (length 4)")

    @property
    def force(self):
        (force, torque) = pwm_to_force_torque(np.array(self._pwm),
                                              self.L,
                                              self.C_F,
                                              self.C_T)
        return rigid.Vector(x=0, y=0, z=-force, reference=self)
    @force.setter
    def force(self, value):
        torque = self.torque

        if not isinstance(value, rigid.Vector):
            value = rigid.Vector(z=-float(value), reference=self)

        self._pwm = tuple(np.squeeze(force_torque_to_pwm(-value.z,
                          self.torque.array,
                          self.L,
                          self.C_F,
                          self.C_T)))

    @property
    def force_local(self):
        return self.rotation*self.force
    @force_local.setter
    def force_local(self, value):
        if isinstance(value, np.ndarray):
            force = self.rotation.inv() * \
                        rigid.Vector(*value, reference=self.reference)
            self.force = force.z
        else:
            raise Exception("Invalid argument: " + str(value) +  "\n" + \
                            "Valid types: 3x1 numpy.array")


    @property
    def torque(self):
        (force, torque) = pwm_to_force_torque(np.array(self._pwm),
                                              self.L,
                                              self.C_F,
                                              self.C_T)
        return rigid.Vector(*torque, reference=self)
    @torque.setter
    def torque(self, value):
        force = self.force

        if isinstance(value, np.ndarray) or isinstance(value, tuple):
            torque = rigid.Vector(*value, reference=self)

        self._pwm = tuple(np.squeeze(force_torque_to_pwm(-force.z,
                                                         torque.array,
                                                         self.L,
                                                         self.C_F,
                                                         self.C_T)))

    @property
    def torque_attitude(self):
        return tuple(np.squeeze(mechanics.ang_torque_to_att_torque(self.attitude, self.torque.array)))
    @torque_attitude.setter
    def torque_attitude(self, value):
        self.torque = mechanics.att_torque_to_ang_torque(self.attitude, np.array([value]).T)

    @property
    def input(self):
        return np.vstack([-self.force.z,
                         np.array([self.torque_attitude]).T])
    @input.setter
    def input(self, value):
        self.force = -float(value[0])
        self.torque_attitude = tuple(np.squeeze(value[1:4]))

    @property
    def state_dot(self):
        output = self.dynamics()
        return output[0]

    @property
    def A(self):
        output = self.dynamics()
        return output[1]

    @property
    def B(self):
        output = self.dynamics()
        return output[2]

    @property
    def A_trans(self):
        output = self.dynamics()
        return output[3]

    @property
    def B_trans(self):
        output = self.dynamics()
        return output[4]

    @property
    def A_rot(self):
        output = self.dynamics()
        return output[5]

    @property
    def B_rot(self):
        output = self.dynamics()
        return output[6]


    @property
    def frame(self):  # pylint: disable=C0111
        return frames.AircraftFrame(name=self.name,
                                    position=self.xyz, rotation=self.quat,
                                    reference=self.reference)
    # Kinetic energy
    @property
    def kinetic(self):  # pylint: disable=C0111
        return mechanics.kinetic_energy(self.M, self.I,
                                        self.linear_vel.array,
                                        self.angular_vel.array)

    # Potential energy
    @property
    def potential(self):  # pylint: disable=C0111
        return mechanics.potential_energy(self.M, self.G, self.z)

    # Total energy
    @property
    def energy(self):  # pylint: disable=C0111
        return self.kinetic + self.potential

    # Lagragian
    @property
    def lagragian(self):  # pylint: disable=C0111
        return self.kinetic - self.potential

    # Coriolis
    @property
    def coriolis(self):  # pylint: disable=C0111
        (phi, theta, psi) = self.attitude
        (phi_dot, theta_dot, psi_dot) = self.attitude_vel
        return quad_coriolis(self.I, phi, theta, phi_dot, theta_dot, psi_dot)

    # Overloaded operators
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return_str = ""

        if self.name is not None:
            return_str += "Name: '" + self.name + "'\n"

        if self.reference is not None:
            if self.reference.name is not None:
                ref_name = self.reference.name
            else:
                ref_name = ""
            return_str += "Reference: '" + ref_name + "'\n"

        if self.t is not None:
            return_str += "Time: " + self.val2str(self.t) + "\n"

        state_dot = self.state_dot
        return_str += "PWM: " + self.val2str(self.pwm) + "\n"
        return_str += "Force (quad frame):   " + self.val2str(self.force.xyz) + "\n"
        return_str += "Force (local frame):  " + self.val2str(self.force_local.xyz) + "\n"
        return_str += "Torque (quad frame):  " + self.val2str(self.torque.xyz) + "\n"
        return_str += "Torque (Euler):  " + self.val2str(tuple(np.squeeze(mechanics.ang_torque_to_att_torque(self.attitude, self.torque.array)))) + "\n"
        return_str += "Position:     " + self.val2str(self.xyz) + "\n"
        return_str += "Attitude:     " + self.val2str(self.attitude) + "\n"
        return_str += "Position Vel: " + self.val2str(self.linear_vel.xyz) + "\n"
        return_str += "Attitude Vel: " + self.val2str(self.attitude_vel) + "\n"
        return_str += "Position Acc: " + self.val2str(tuple(np.squeeze(state_dot[6:9]))) + "\n"
        return_str += "Attitude Acc: " + self.val2str(tuple(np.squeeze(state_dot[9:12]))) + "\n"

        return return_str


    # Methods
    def finite_state_update(self):
        if self.finite_state[0] == 'LANDED':
            if not self.at_state_desired:
                self.finite_state[0] = 'MOVING'

        elif self.finite_state[0] == 'MOVING':
            if self.at_state_desired:
                if self.state_desired[2, 0] <= 0:
                    self.finite_state[0] = 'LANDED'
                else:
                    self.finite_state[0] = 'HOLDING'

        elif self.finite_state[0] == 'HOLDING':
            if not self.at_state_desired:
                self.finite_state[0] = 'MOVING'

    def update(self, ts):
        u_plus = self.controller()
        self.input = u_plus

        self.t = self.t + ts
        state = self.state + self.state_dot*ts
        state[6:9] = np.maximum(np.minimum(state[6:9],
                                self._max_lin_vel), -self._max_lin_vel)
        state[9:12] = np.maximum(np.minimum(state[9:12],
                                self._max_ang_vel), -self._max_ang_vel)
        self.state = state
        self._state_err_cum_sum = self._state_err_cum_sum + \
                                  (self.state_desired - self.state) * ts

        self.finite_state_update()


    def f(self, x, u):
        """
        @type x: 12x1 numpy.array
        @param x: State vector [x, y, z,
                                roll, pitch, yaw,
                                x_vel, y_vel, z_vel,
                                roll_vel, pitch_vel, yaw_vel]

        @type u: 6x1 numpy.array
        @param u: Input vector [force_x, force_y, force_z
                                torque_roll, torque_pitch, torque_yaw]
                  Note force is in the quad frame and torque is attitude
                  (or Euler) torques.
        """
        attitude = tuple(np.squeeze(x[3:6]))
        linear_vel = rigid.Vector(*x[6:9], reference=self.reference)
        attitude_vel = tuple(np.squeeze(x[9:12]))
        force = u[0:3]
        torque_attitude = u[3:6]
        output = self.dynamics(attitude=attitude,
                               linear_vel=linear_vel,
                               attitude_vel=attitude_vel,
                               force=force,
                               torque_attitude=torque_attitude)
        x_dot = output[0]
        return x_dot

    def dynamics(self,
                 attitude=None,
                 linear_vel=None,
                 attitude_vel=None,
                 pwm=None,
                 force=None,
                 torque=None,
                 torque_attitude=None):

        if attitude is None:
            attitude = self.attitude
        if linear_vel is None:
            linear_vel = self.linear_vel
        if attitude_vel is None:
            attitude_vel = self.attitude_vel

        if pwm is None:
            if force is None:
                force = self.force
            if torque is None and torque_attitude is None:
                torque_attitude = np.array([self.torque_attitude]).T
            elif torque is not None and torque_attitude is None:
                torque_attitude = mechanics.ang_torque_to_att_torque(self.attitude, torque.array)
            elif torque is not None and torque_attitude is not None:
                raise Exception("Invalid argument. Either use torque or torque_attitude, but not both.")

        elif pwm is not None and force is None and torque is None:
            (force, torque) = pwm_to_force_torque(np.array(pwm),
                                                  self.L,
                                                  self.C_F,
                                                  self.C_T)
            # force_local = self.rotation * rigid.Vector(x=0, y=0, z=force, reference=self)
            force = rigid.Vector(x=0, y=0, z=-force, reference=self)
            torque = rigid.Vector(*torque, reference=self)
            torque_attitude = mechanics.ang_torque_to_att_torque(self.attitude, torque.array)
        else:
            raise Exception("Invalid arguments. Either use pwm or (force and torque), but not both.")

        (x_dot_trans, x_dot_rot, A_trans, B_trans, A_rot, B_rot) = \
        quad_model.quad_dynamics(self.G, self.M,
                self.I[0, 0], self.I[1, 1], self.I[2, 2],
                linear_vel.x, linear_vel.y, linear_vel.z,
                attitude[0], attitude[1], attitude[2],
                attitude_vel[0], attitude_vel[1], attitude_vel[2],
                -force.z,
                torque_attitude[0, 0], torque_attitude[1, 0], torque_attitude[2, 0])

        # Rearrange the ordering of the states from
        # [pos pos_vel att att_vel] to [pos, att, pos_vel, att_vel]
        i3 = np.identity(3)
        z3 = np.zeros((3, 3))

        # Note: For this similarity transform T_inv = T.T = T
        T_inv = np.vstack([np.hstack([i3, z3, z3, z3]),
                          np.hstack([z3, z3, i3, z3]),
                          np.hstack([z3, i3, z3, z3]),
                          np.hstack([z3, z3, z3, i3])])

        f_bar = np.vstack([x_dot_trans, x_dot_rot])
        x_dot = np.dot(T_inv, f_bar)

        A_left = np.vstack([A_trans, np.zeros((6, 6))])
        A_right = np.vstack([np.zeros((6, 6)), A_rot])
        A_bar = np.hstack([A_left, A_right])
        A = np.dot(T_inv, np.dot(A_bar, T_inv))

        B_top = np.hstack([B_trans, np.zeros((6, 3))])
        B_bot = np.hstack([np.zeros((6, 1)), B_rot])
        B_bar = np.vstack([B_top, B_bot])
        B = np.dot(T_inv, B_bar)

        # Temporary
        # x_dot[0:3] = np.zeros((3,1))

        if self.finite_state[0] == 'LANDED':
            x_dot[2, 0] = max(x_dot[2, 0], 0)

        return (x_dot, A, B, A_trans, B_trans, A_rot, B_rot)

    # Methods -- Controllers
    def zero_control(self):
        return np.zeros((self.input_dim, 1))

    def pass_through_control(self):
        return self.input

    def pid_control(self):
        u_bar = self.input_desired

        # e = self.state_err[0:6]
        e_pos = self.rotation.inv()*rigid.Vector(*self.state_err[0:3], reference=self.reference)
        e_vel = self.rotation.inv()*rigid.Vector(*self.state_err[3:6], reference=self.reference)
        e = np.vstack([e_pos.array, e_vel.array])

        # e_i = self.state_err_cum_sum[0:6]
        e_i_pos = self.rotation.inv()*rigid.Vector(*self.state_err_cum_sum[0:3], reference=self.reference)
        e_i_vel = self.rotation.inv()*rigid.Vector(*self.state_err_cum_sum[3:6], reference=self.reference)
        e_i = np.vstack([e_i_pos.array, e_i_vel.array])

        # e_d = self.state_err[6:12]
        e_d_pos = self.rotation.inv()*rigid.Vector(*self.state_err[6:9], reference=self.reference)
        e_d_vel = self.rotation.inv()*rigid.Vector(*self.state_err[9:12], reference=self.reference)
        e_d = np.vstack([e_d_pos.array, e_d_vel.array])

        P = self.pid_P
        I = self.pid_I
        D = self.pid_D

        u_delta = np.dot(P, e) + np.dot(I, e_i) + np.dot(D, e_d)

        return u_bar + u_delta

    def lqr_attitude_control(self):
        Q_rot = np.diag([80, 80, 2, 1, 1, 1])
        R_rot = np.diag(np.array([1000, 1000, 1000]))
        K_rot,_,_ = lqr(self.A_rot, self.B_rot, Q_rot, R_rot)
        X_rot = np.vstack([self.state[3:6], self.state[9:12]])

        torque_attitude = -np.dot(K_rot, X_rot)

        force = self.G*self.M

        return np.vstack([force, torque_attitude])

    def lqr_position_attitude_control(self):
        m = self.M
        g = self.G
        q = self.state

        xi = q[0:3]
        xi_dot = q[6:9]
        phi = q[3,0]
        theta = q[4,0]
        psi = mmath.wrap_to_2pi(q[5,0])
        eta = np.array([[phi, theta, psi]]).T
        chi = np.vstack([xi, xi_dot])

        xi_bar = self.state_desired[0:3]
        psi_bar = mmath.wrap_to_2pi(self.state_desired[5,0])
        xi_dot_bar = self.state_desired[6:9]
        chi_bar = np.vstack([xi_bar, xi_dot_bar])

        chi_delta = (chi - chi_bar)

        A_pos = np.vstack([np.hstack([np.zeros((3,3)), np.identity(3)]),
                           np.zeros((3, 6))])

        B_pos = np.vstack([np.zeros((3, 3)), np.identity(3)])
        Q_pos = np.diag([2, 2, 2, 20, 20, 20])
        R_pos = np.diag(np.array([5, 5, 5]))
        K_pos,_,_ = lqr(A_pos, B_pos, Q_pos, R_pos)

        acc_vec = -np.dot(K_pos, chi_delta) + np.array([[0, 0, g]]).T

        Z = -acc_vec / np.linalg.norm(acc_vec)
        a = Z[0, 0]*sin(psi_bar) + Z[1, 0]*cos(psi_bar)
        b = -Z[2, 0]
        c = -a

        if np.allclose(a, 0):
            X3 = 0
        elif abs((-b+sqrt(b**2 - 4*a*c))/(2*a)) <= 1:
            X3 = (-b+sqrt(b**2 - 4*a*c))/(2*a)
        else:
            X3 = (-b-sqrt(b**2 - 4*a*c))/(2*a)

        d = sqrt(1 - X3**2)

        if psi_bar >= 0 and psi_bar < PI:
            X1 = sqrt(d**2 * (sin(psi_bar))**2)
        else:
            X1 = -sqrt(d**2 * (sin(psi_bar))**2)
        if (psi_bar >= 0 and psi_bar < PI/2) or psi_bar >= 3*PI/2:
            X2 = sqrt(d**2 * (cos(psi_bar))**2)
        else:
            X2 = -sqrt(d**2 * (cos(psi_bar))**2)

        X = np.array([[X1, X2, X3]]).T
        X = X / np.linalg.norm(X)
        Y = np.atleast_2d(np.cross(np.squeeze(Z), np.squeeze(X))).T
        Y = Y / np.linalg.norm(Y)

        R = np.hstack([X, Y, Z])

        (phi_bar, theta_bar, _) = mmath.rot2euler(np.dot(self.nominal_frame.rot_mat, R))
        eta_bar = np.array([[phi_bar, theta_bar, psi_bar]]).T
        eta_delta = eta - eta_bar
        eta_delta = np.array([[mmath.wrap_to_pi(ang) for ang in eta_delta]]).T


        Q_rot = np.diag([1, 1, 1, 1, 1, 1])
        R_rot = np.diag(np.array([5, 5, 5]))
        K_rot,_,_ = lqr(self.A_rot, self.B_rot, Q_rot, R_rot)
        X_rot = np.vstack([eta_delta, self.state[9:12]])
        torque_attitude = -np.dot(K_rot, X_rot)

        force = np.linalg.norm(acc_vec)*m

        u = np.vstack([force, torque_attitude])

        return u

    def clf_attitude_control(self):
        eta_bar = self.state_desired[3:6]
        (phi, theta, psi) = self.attitude
        eta = np.atleast_2d(np.array((phi, theta, psi))).T

        eta_delta = eta - eta_bar
        (phi_dot, theta_dot, psi_dot) = self.attitude_vel
        eta_dot = np.atleast_2d(np.array((phi_dot, theta_dot, psi_dot))).T
        J = generialized_coordinate_inertia(self.I, phi, theta, psi)
        C = quad_coriolis(self.I, phi, theta, phi_dot, theta_dot, psi_dot)

        eps = 5
        u_hat = -(eps*eta_delta + 2*eta_dot)
        torque_attitude = np.dot(J, u_hat) + np.dot(C, eta_dot)

        # force = np.array([[0, 0, self.G*self.M / self.rot_mat[2, 2]]]).T
        force = self.G*self.M
        # force = 0

        return np.vstack([force, torque_attitude])

    def diffeomorphism_full_state_control(self):
        d = 1
        xi_bar = np.array([[0, 0, 1]]).T

        zeta_bar = xi_bar + np.array([[0, 0, d]]).T

        m = self.M
        g = self.G
        R = self.rot_mat
        x = self.state
        xi = x[0:3]
        eta = x[3:6]
        xi_dot = x[6:9]
        eta_dot = x[9:12]
        phi = eta[0]
        theta = eta[1]
        psi = eta[2]
        phi_dot = eta_dot[0]
        theta_dot = eta_dot[1]
        psi_dot = eta_dot[2]

        zeta = (self * rigid.Vector(*(0, 0, -d))).array
        chi = np.vstack([zeta-zeta_bar, psi, xi_dot, psi_dot])
        W = mechanics.att_vel_to_ang_vel_jacobian(phi, theta)
        W_inv = np.linalg.inv(W)
        W_dot = mechanics.time_derivative_att_vel_to_ang_vel_jacobian(phi,
                                                                      theta,
                                                                      phi_dot,
                                                                      theta_dot)



        A = np.hstack([np.zeros((8, 4)),
                      np.vstack([np.identity(4), np.zeros((4, 4))])])
        b = np.array([[-1/m*R[0, 2], d*R[0, 1], -d*R[0, 0]],
                     [-1/m*R[1, 2], d*R[1, 1], -d*R[1, 0]],
                     [-1/m*R[2, 2], d*R[2, 1], -d*R[2, 0]]])
        B = np.vstack([np.zeros((4, 4)),
                      np.hstack([b, np.zeros((3, 1))]),
                      np.array([[0, 0, 0, 1]])])
        G = np.array([[0, 0, 0, 0, 0, 0, -g, 0]]).T

        QQ = np.diag([100, 100, 10, 1, 1, 1, 1, 1])
        RR = np.diag(np.array([1, 1, 1, 1]))
        K,_,_ = lqr(A, B, QQ, RR)
        # mu_bar = np.array([[-g*m*R[2, 2], g/d*R[2, 1], -g/d*R[2, 0], 0]]).T
        mu_bar = np.array([[g*m, 0, 0, 0]]).T
        mu_delta = -np.dot(K, chi)
        mu = mu_bar + mu_delta

        F = mu[0, 0]
        alpha = mu[1:4]

        J = generialized_coordinate_inertia(self.I, phi, theta, psi)
        C = quad_coriolis(self.I, phi, theta, phi_dot, theta_dot, psi_dot)
        tau_eta = np.dot(J, np.dot(W_inv, (alpha - np.dot(W_dot, eta_dot)))) + np.dot(C, eta_dot)

        return np.vstack([F, tau_eta])


def pwm_to_force_torque(pwm, L, C_F, C_T):
    if isinstance(pwm, tuple):
        pwm = np.array(pwm)
    if len(pwm) != 4:
        raise Exception("Invalid argument:\n" + str(pwm) + \
                    "\n" + "Valid types: 1x4 numpy.ndarray")

    force_motor = C_F * pwm**2
    torque_motor = C_T * pwm**2

    F1 = float(force_motor[0])
    F2 = float(force_motor[1])
    F3 = float(force_motor[2])
    F4 = float(force_motor[3])
    T1 = float(torque_motor[0])
    T2 = float(torque_motor[1])
    T3 = float(torque_motor[2])
    T4 = float(torque_motor[3])

    force = np.sum(force_motor)
    torque = np.array([L/sqrt(2)*(F2 + F3 - F1 - F4), L/sqrt(2)*(F1 + F3 - F2 - F4), T1 + T2 - T3 - T4])

    return (force, torque)

def force_torque_to_pwm(force, torque, L, C_F, C_T):

    F = float(force)
    Tx = float(torque[0])
    Ty = float(torque[1])
    Tz = float(torque[2])

    C_F1 = C_F[0]
    C_F2 = C_F[1]
    C_F3 = C_F[2]
    C_F4 = C_F[3]

    C_T1 = C_T[0]
    C_T2 = C_T[1]
    C_T3 = C_T[2]
    C_T4 = C_T[3]

    pwm1_sq = (C_F2*C_F3*C_T4*F*L + C_F2*C_F4*C_T3*F*L + 2*C_F2*C_F3*C_F4*L*Tz - 2**(.5)*C_F2*C_F3*C_T4*Tx - 2**(.5)*C_F3*C_F4*C_T2*Tx + 2**(.5)*C_F2*C_F4*C_T3*Ty + 2**(.5)*C_F3*C_F4*C_T2*Ty)/(2*L*(C_F1*C_F2*C_F3*C_T4 + C_F1*C_F2*C_F4*C_T3 + C_F1*C_F3*C_F4*C_T2 + C_F2*C_F3*C_F4*C_T1))
    pwm2_sq = (2**(.5)*(2*C_F1*C_F4*C_T3*Tx + 2*C_F3*C_F4*C_T1*Tx - 2*C_F1*C_F3*C_T4*Ty - 2*C_F3*C_F4*C_T1*Ty + 2**(.5)*C_F1*C_F3*C_T4*F*L + 2**(.5)*C_F1*C_F4*C_T3*F*L + 2*2**(.5)*C_F1*C_F3*C_F4*L*Tz))/(4*L*(C_F1*C_F2*C_F3*C_T4 + C_F1*C_F2*C_F4*C_T3 + C_F1*C_F3*C_F4*C_T2 + C_F2*C_F3*C_F4*C_T1))
    pwm3_sq = (2**(.5)*(2*C_F1*C_F2*C_T4*Tx + 2*C_F1*C_F4*C_T2*Tx + 2*C_F1*C_F2*C_T4*Ty + 2*C_F2*C_F4*C_T1*Ty + 2**(.5)*C_F1*C_F4*C_T2*F*L + 2**(.5)*C_F2*C_F4*C_T1*F*L - 2*2**(.5)*C_F1*C_F2*C_F4*L*Tz))/(4*L*(C_F1*C_F2*C_F3*C_T4 + C_F1*C_F2*C_F4*C_T3 + C_F1*C_F3*C_F4*C_T2 + C_F2*C_F3*C_F4*C_T1))
    pwm4_sq = -(2*C_F1*C_F2*C_F3*L*Tz - C_F2*C_F3*C_T1*F*L - C_F1*C_F3*C_T2*F*L + 2**(.5)*C_F1*C_F2*C_T3*Tx + 2**(.5)*C_F2*C_F3*C_T1*Tx + 2**(.5)*C_F1*C_F2*C_T3*Ty + 2**(.5)*C_F1*C_F3*C_T2*Ty)/(2*L*(C_F1*C_F2*C_F3*C_T4 + C_F1*C_F2*C_F4*C_T3 + C_F1*C_F3*C_F4*C_T2 + C_F2*C_F3*C_F4*C_T1))

    pwm1 = min(max(np.sqrt(abs(pwm1_sq)), 0), 100)
    pwm2 = min(max(np.sqrt(abs(pwm2_sq)), 0), 100)
    pwm3 = min(max(np.sqrt(abs(pwm3_sq)), 0), 100)
    pwm4 = min(max(np.sqrt(abs(pwm4_sq)), 0), 100)

    return np.array([pwm1, pwm2, pwm3, pwm4])

def generialized_coordinate_inertia(I, phi, theta, psi):
    W = mechanics.att_vel_to_ang_vel_jacobian(phi, theta)
    J = np.dot(W.T, np.dot(I, W))
    return J

def time_derivative_generialized_coordinate_inertia(I, phi, theta, phi_dot, theta_dot):
    W = mechanics.att_vel_to_ang_vel_jacobian(phi, theta)
    W_dot = mechanics.time_derivative_att_vel_to_ang_vel_jacobian(phi, theta, phi_dot, theta_dot)
    J_dot = 2 * np.dot(W.T, np.dot(I, W_dot))
    return J_dot

def quad_coriolis(I, phi, theta, phi_dot, theta_dot, psi_dot):
    J_dot = time_derivative_generialized_coordinate_inertia(I, phi, theta, phi_dot, theta_dot)

    I_x = I[0, 0]
    I_y = I[1, 1]
    I_z = I[2, 2]

    dM_11 = 0.
    dM_21 = -I_x*cos(theta)*psi_dot
    dM_31 = 0.

    dM_12 = -(I_y - I_z)*(theta_dot*sin(2*phi) + psi_dot*cos(theta) - 2*psi_dot*cos(phi)**2*cos(theta))
    dM_22 = -(I_y - I_z)*cos(phi)*sin(phi)*sin(theta)*psi_dot
    dM_32 = 0.

    dM_13 = cos(theta)*(I_y - I_z)*(2*theta_dot*cos(phi)**2 - theta_dot + 2*psi_dot*cos(phi)*cos(theta)*sin(phi))
    dM_23 = -theta_dot*(I_y-I_z)*cos(phi)*sin(phi)*sin(theta) - psi_dot*(2*I_z*cos(phi)**2*cos(theta)*sin(theta) - 2*I_x*cos(theta)*sin(theta) + 2*I_y*cos(theta)*sin(phi)**2*sin(theta)) - I_x*phi_dot*cos(theta)
    dM_33 = 0.

    dM_dEta = np.array([[dM_11, dM_12, dM_13],
                        [dM_21, dM_22, dM_23],
                        [dM_31, dM_32, dM_33]])

    C = J_dot - 0.5 * dM_dEta

    return C

def quad_dynamics(g, m, I, phi, theta, psi, phi_dot, theta_dot, psi_dot, force, torque):
    R_nominal = np.array([[0, 1, 0],
                          [1, 0, 0],
                          [0, 0, -1]])

    R_psi = lambda psi_: np.array([[cos(psi_), -sin(psi_), 0],
                                   [sin(psi_), cos(psi_), 0],
                                   [0, 0, 1]])

    R_theta = lambda theta_: np.array([[cos(theta_), 0, sin(theta_)],
                                       [0, 1, 0],
                                       [-sin(theta_), 0, cos(theta_)]])

    R_phi = lambda phi_: np.array([[1, 0, 0],
                                   [0, cos(phi_), -sin(phi_)],
                                   [0, sin(phi_), cos(phi_)]])

    R_L_Q= lambda phi_, theta_, psi_: np.dot(R_nominal, np.dot(R_psi(psi_), np.dot(R_theta(theta_), R_phi(phi_))))

    a_L = np.dot(R_L_Q(phi, theta, psi), force.array) / m
    x_ddot = float(a_L[0])
    y_ddot = float(a_L[1])
    z_ddot = float(a_L[2] - g)

    J = generialized_coordinate_inertia(I, phi, theta, psi)
    J_inv = np.linalg.inv(J)
    C = quad_coriolis(I, phi, theta, phi_dot, theta_dot, psi_dot)
    eta_dot = np.array([[phi_dot], [theta_dot], [psi_dot]])

    V = torque.array - np.dot(C, eta_dot)

    eta_ddot = np.dot(J_inv, V)

    phi_ddot = float(eta_ddot[0])
    theta_ddot = float(eta_ddot[1])
    psi_ddot = float(eta_ddot[2])

    return (x_ddot, y_ddot, z_ddot, phi_ddot, theta_ddot, psi_ddot)
