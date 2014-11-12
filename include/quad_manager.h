/**
 * @brief Quadrotor manager class
 *
 * @file quad_manager.h
 * @author Rowland O'Flaherty
 * @date 11/12/2014
 **/

#ifndef QUAD_MANAGER_H
#define QUAD_MANAGER_H

//------------------------------------------------------------------------------
// Includes
//------------------------------------------------------------------------------
#include "quad.h"

#include <vector>

//------------------------------------------------------------------------------
// Namespaces
//------------------------------------------------------------------------------
using namespace std;

//==============================================================================
// QuadManager Class
//==============================================================================
class QuadManager
{
public:
    //--------------------------------------------------------------------------
    // Constructors and Destructors
    //--------------------------------------------------------------------------
    QuadManager(int n_quads=1);

    // ~QuadManager();

    //--------------------------------------------------------------------------
    // Public Member Getters and Setters
    //--------------------------------------------------------------------------
    int n_quads const { return m_n_quads; }



    //--------------------------------------------------------------------------
    // Public Member Variables
    //--------------------------------------------------------------------------


    //--------------------------------------------------------------------------
    // Public Methods
    //--------------------------------------------------------------------------

private:
    //--------------------------------------------------------------------------
    // Private Member Variables
    //--------------------------------------------------------------------------
    int m_n_quads;
    vector<Quad> m_quad;

    //--------------------------------------------------------------------------
    // Private Methods
    //--------------------------------------------------------------------------

};

#endif
