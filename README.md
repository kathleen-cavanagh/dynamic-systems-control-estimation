# dynamic-systems-control-estimation
Adventures in applying control and estimation techniques to dynamic systems.

This project is the continuation of work developed in 6.832 (Underactuated Robotics). Using simple systems such as a cart pole or acrobot (soon to come!), traditional control and estimation techniques are implemented to understand behavior and trade offs more intuitively. 

The structure is as follows
- Simple example scripts are provided in `dev`
- Dynamic systems and trajectory optimizations are defined in the `systems` folder. Currently only the Cart Pole system and two types of measurements on the system have been implemented. There is support for direct transcription and direct collocation trajectory optimization built using pydrake (https://drake.mit.edu).
- Estimation capabilities are found in `estimation`. Currently an EKF and a UKF have been implemented
- The `utils.py` file contains utilities used across the codebase as well as a preliminary implementation of time varying LQR which will be further developed at a later point in time.

This is a personal project so please excuse the lack of unit-testing etc.
