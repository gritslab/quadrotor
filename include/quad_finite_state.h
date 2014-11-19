/*
@brief Enums and class defining the quad's finite states

@author Rowland O'Flaherty
@date 11/12/2014
*/

#ifndef QUAD_FINITE_STATES_H
#define QUAD_FINITE_STATES_H

//------------------------------------------------------------------------------
// Includes
//------------------------------------------------------------------------------
#include <cassert>

//------------------------------------------------------------------------------
// Constants and Enums
//------------------------------------------------------------------------------

// Finite States
const int FINITE_STATE_DIM = 3;
const int FINITE_STATE_DIM_SIZE [FINITE_STATE_DIM] = {2, 4, 2};

// DIM 0
const int ARM_STATE = 0;
enum ArmState
{
    DISARMED=0,
    ARMED
};

// DIM 1
const int FLIGHT_STATE = 1;
enum FlightState
{
    GROUNDED=0,
    HOLDING,
    AUTONOMOUS,
    TELEOP
};

// DIM 2
const int SETPOINT_STATE = 2;
enum SetpointState
{
    AT_SETPOINT = 0,
    GOINT_TO_SETPOINT
};

// Transitions
const int NUM_OF_COMMANDS = 4;
enum Commands
{
    TOGGLE_ARM=0,
    TAKEOFF,
    FLY_TO,
    LAND,
    CALIBRATE,
    RETURN,
    CHANGE_MODE
};


//==============================================================================
// QuadFiniteState Class
//==============================================================================
class QuadFiniteState
{
private:
    int m_finite_state[FINITE_STATE_DIM];

public:
    int& operator[] (const int n_index) {
        assert(n_index >= 0 && n_index < FINITE_STATE_DIM);
        return m_finite_state[n_index];
    }
};

#endif
