# Real-World MuSHR Platform Controller

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)]()
[![Ubuntu](https://img.shields.io/badge/Ubuntu-20.04-purple.svg)]()

</div>

[Easop Lee](https://easoplee.github.io/),
[Samuel A. Moore](https://samavmoore.github.io/), and
[Boyuan Chen](http://boyuanchen.com/)
<br>
Duke University
<br>


> Offboard control of sym2real on [MuSHR](https://mushr.io/) racecar w/ vicon mocap data

---

⚠️ **Important Notice**  
This repository **must be used together** with the [Sym2Real](https://github.com/easoplee/sym2real) framework. Please check out this repository first.

- The **Sym2Real repo** provides modular dynamics models, controllers, and simulation environments.  
- This **MuSHR controller repo** provides the hardware interface for the racecar real-world experiments.  

Please **keep both repositories in sync** when running experiments. Updates or mismatches between the two may cause unexpected errors.

---

## 🔗 Syncing with Sym2Real

Make sure you clone the [Sym2Real](https://github.com/easoplee/sym2real) repo, and follow its installation instructions first.

Then, clone this repository **next to** the [Sym2Real](https://github.com/easoplee/sym2real) (e.g. if the parent folder is Desktop):

```
Desktop/
├── sym2real
└── mushr-controller
```

```
git clone git@github.com:easoplee/mushr-controller.git
```

## 🛠️ Installation

Connect to the the same wifi (e.g., VICON_Server_5G) for both the Jetson on mushr and your Dekstop.

```
- Robot IP: ssh el183@192.168.0.x
- Desktop IP: 192.168.0.y
```

ssh into the robot Jetson, and update the last line in ~/.bashrc file
```
export ROS_IP=192.168.0.x
```

In your offboard PC, run the following:
```
source setup.sh # make sure ROS_IP matches hostname -I on Desktop
```
but make sure setup.sh ip addresses reflect the correct robot and Desktop IP.

## ✅ Sanity Check

First, run ```roslaunch mushr_base teleop.launch``` on Robot Jetson. To test ackermann command (steering angle, velocity) publishing script, run the following script on your Desktop.

```
python tests/simple_send_commands.py
```

- If robot does not respond to ```python send_commands.py```, disable firewall on Desktop as follows:
```sudo ufw disable```

To stream Vicon data, launch the Vicon launch script from your Desktop. Once it’s running, test reading and sending data using the following script. Our script assumes the Vicon object name is "mushr".

⚠️ Make sure the car has enough space to move. It should move randomly.

Quit the script once this is verified.

```
python tests/read_send.py
```

## 🚀 Quickstart

To run what would happen in the simulation:
```
python sim_preview.py
```

▶️ Run Base Model (e.g. Symbolic Regression in the real world, zero-shot):
```
python main.py \
    dynamics_model="symbolic_regression" \
    model_path="{path_to_sr_base_model_experiment}" \
    overrides.reference_type="circle"
```
```
e.g., model_path="/home/generalroboticslab/Desktop/sym2real/example_results/MushrCar-v0/hover/env_params_mass_0.027_wheel_friction_2.0_floor_friction_0.5/seed_1/symbolic_regression/2025.08.06:040815"

```

Then, to train the residual model on the collected real world data: 
```
python train.py \
    dynamics_model="residual_mlp"
    dynamics_model.model_a="symbolic_regression" \
    dynamics_model.model_a_path="{path_to_sr_base_model_experiment}" \
    overrides.reference_type="circle"
```
```
e.g., dynamics_model.model_a_path="/home/generalroboticslab/Desktop/sym2real/example_results/MushrCar-v0/hover/env_params_mass_0.027_wheel_friction_2.0_floor_friction_0.5/seed_1/symbolic_regression/2025.08.06:040815"
```

🪄 Run Residual Model - Finetuned (e.g. Symbolic Regression + Residual MLP):
```
python main.py \
    dynamics_model="residual_mlp" \
    dynamics_model.model_a="symbolic_regression" \
    dynamics_model.model_a_path="{path_to_sr_base_model_experiment}" \
    model_path="{path_to_residual_mlp_model_experiment}" \
    overrides.reference_type="circle"
```
```
model_path="/home/generalroboticslab/Desktop/mushr_controller/real_exp_results/offset_0.1_sr_mlp/circle/seed=1/dynamics_model/mlp_model_w_dataset_size_400"
```
