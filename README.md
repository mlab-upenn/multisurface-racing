# multisurface-racing

## Simulator
This repository is using multi-body full scale car version of the [F1/10 gym](https://github.com/atomyks/f1tenth_gym/tree/multibody).

## Entry points
* GP_MPC.py
    * MPC controller that is using learned GP model.
* LMPC.py
    * Currently does not work (Work in progress).
    * Will be used for comparison as a different model learning method.
    * Atempt to run [LMPC](https://github.com/urosolia/Learning_Robust_MPC/blob/main/Nominal_LMPC_Chapter/Ex3_LMPC_for_autonomous_racing/Ex3_LMPC_for_autonomous_racing.ipynb) in F1/10 gym.
    * Currently can drive pure pursuit for the first three laps and create a "safe set" for LMPC initialization.
* Kinematic_MPC.py
    * You can choose from kinematin MPC or pure pursuit controler.
    * Currently used for comparison against GP-MPC.
