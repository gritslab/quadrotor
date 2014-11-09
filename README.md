# quadrotor

GRITS Lab ROS quadrotor package

## Prerequisite

[python-control package](http://python-control.sourceforge.net/manual-0.5a/)

0. Install [Slycot](https://github.com/repagh/Slycot)

    ```bash
    sudo apt-get install python-scipy
    sudo apt-get install pip
    sudo pip install slycot
    ```

0. Download python-control package from http://sourceforge.net/p/python-control/wiki/Download/

    ```bash
    cd ~/Downloads
    tar -xzf control-0.6d.tar.gz
    cd control-0.6d.tar.gz
    python setup.py install
    ```

## Running
To run the quadrotor simulator do:

    ```bash
    roslaunch quadrotor quad_sim.launch
    ```
