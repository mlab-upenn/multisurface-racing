# multisurface-racing

## Simulator
This repository is using multi-body full scale car version of the [F1/10 gym](https://github.com/atomyks/f1tenth_gym/tree/multibody).

## Entry points
* GP_MPC.py
    * Runs the first lap using a standard controller (Extended-Kinematic MPC or Pure Pursuit) and collects data for GP training.
    * Trains the GP using the collected dataset and then switches to GP MPC.
    * After every lap, it saves the dataset generated from driving on the track to a file called "testing_dataset.json".
* GP_MPC_eval.py
    * Loads in the dataset created by GP_MPC.py and retrains the GP on this dataset.
    * The trained GP(s) are then evaluated for prediction error.
* Kinematic_and_dynamic_MPC.py.py
    * You can choose from extended-kinematic MPC, dynamic (bicycle model) MPC, and pure pursuit controler.
    * Currently used for comparison against GP-MPC.

## Citing
If you find this Controller useful, please consider citing:

```
@article{nagy2023ensemble,
  title={Ensemble Gaussian Processes for Adaptive Autonomous Driving on Multi-friction Surfaces},
  author={Nagy, Tom{\'a}{\v{s}} and Amine, Ahmad and Nghiem, Truong X and Rosolia, Ugo and Zang, Zirui and Mangharam, Rahul},
  journal={arXiv preprint arXiv:2303.13694},
  year={2023}
}
```
