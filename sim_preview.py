import os
os.environ["MUJOCO_GL"] = "egl"     

import time
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
from geometry_msgs.msg import TransformStamped

import gymnasium as gym
import sym2real.envs

from sym2real.dynamics_model import SymbolicPysr
from sym2real.dynamics_model import GaussianMLP
from sym2real.controllers.reference_trajectories.setpoint_ref import SetpointReference
from sym2real.controllers.reference_trajectories.mushr_circle_ref import MushrCircleReference
from sym2real.controllers.agent import TrajectoryPlanningAgent
from sym2real.dynamics_model.one_dim_tr_model import OneDTransitionRewardModel
from sym2real.util.math_utils import quat2euler
from sym2real.util.common import create_one_dim_tr_model
from sym2real.controllers.mpc import SplineMPPIController

def get_vicon_pose():
    """Get current pose from Vicon system."""
    pose_data = None
    
    def pose_callback(msg):
        nonlocal pose_data
        pose_data = msg
    
    rospy.init_node("vicon_reader", anonymous=True)
    
    pose_sub = rospy.Subscriber("/vicon/mushr/mushr", TransformStamped, pose_callback)
    
    rospy.loginfo("Waiting for Vicon data...")
    while pose_data is None and not rospy.is_shutdown():
        rospy.sleep(0.1)
    
    if rospy.is_shutdown():
        return None

    x, y = pose_data.transform.translation.x, pose_data.transform.translation.y
    qx, qy, qz, qw = pose_data.transform.rotation.x, pose_data.transform.rotation.y, pose_data.transform.rotation.z, pose_data.transform.rotation.w
    _, _, yaw = quat2euler(np.array([qw, qx, qy, qz]))
    
    rospy.loginfo(f"Vicon pose: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}")
    
    pose_sub.unregister()
    rospy.signal_shutdown("Vicon data received")
    
    return x, y, yaw
    
def main():
    print("Getting initial pose from Vicon...")
    vicon_pose = get_vicon_pose()
    
    if vicon_pose is None:
        print("Failed to get Vicon pose. Exiting.")
        return
    
    x_init, y_init, yaw_init = vicon_pose
    x_init = -0.75
    y_init = 0
    yaw_init = 0
    

    env = gym.make("MushrCar-v0")
    
    dynamics_model = SymbolicPysr(model_type="sym_only",
                                action_type="attitude_rate",
                                deterministic=True,
                                in_size=7+2,
                                out_size=7,
    )
    dynamics_model.load("/home/generalroboticslab/Desktop/sym2real/01_from_scratch_mushr_sr_85/MushrCar-v0/hover/env_params_mass_0.027_wheel_left/seed_1/symbolic_regression/2025.08.06:040815")
    
    dynamics_model = GaussianMLP(in_size=7+2, out_size=7, action_type="attitude_rate")
    dynamics_model.load("/home/generalroboticslab/Desktop/sym2real/01_from_scratch_mushr_sr_85/MushrCar-v0/hover/env_params_mass_0.027_wheel_left/seed_0/gaussian_mlp_ensemble/2025.08.05:210617/dynamics_model/curr")
    
    dynamics_model = OneDTransitionRewardModel(
        model=dynamics_model,
    )
    
    rollout_horizon_in_sec = 1.0
        
    num_samples = 1024

    controller = SplineMPPIController(
        env=env,
        horizon=int(rollout_horizon_in_sec/env.unwrapped.dt),
        dt=env.unwrapped.dt,
        num_samples=num_samples,
        model=dynamics_model,
        seed=0
    )
    
    reference_class = MushrCircleReference(radius=1.0, backwards=False)
    reference_class.set_center([x_init, y_init, 1.0], initial_yaw=yaw_init)
    agent = TrajectoryPlanningAgent(controller,
                                    reference_class=reference_class)
    
    # Warm-up  
    agent.act(t=0, obs=np.zeros((7,)))
    
    xml = "/home/generalroboticslab/Desktop/sym2real/envs/assets/mushr_car/one_car.xml"
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    
    # Set initial position from Vicon
    d.qpos[0] = x_init
    d.qpos[1] = y_init
    
    quat = R.from_euler('xyz', [0, 0, yaw_init]).as_quat()
    quat = np.roll(quat, 1)  # Mujoco uses [w, x, y, z] format
    d.qpos[3:7] = quat

    # Simulation parameters - 20 Hz control frequency
    control_freq = 20.0  # Hz
    control_dt = 1.0 / control_freq  # 0.05 seconds
    sim_time = 20.0  # Total simulation time
    steps = int(sim_time / control_dt)  # Number of control steps
    
    print(f"Starting simulation from Vicon pose: x={x_init:.3f}, y={y_init:.3f}, yaw={yaw_init:.3f}")
    print(f"Goal: x=1.0, y=0.0, yaw=0.0")
    print(f"Running for {sim_time} seconds...")

    corner_1 = np.array([-0.85, -3.8])
    corner_2 = np.array([1.75, -3.8])
    corner_3 = np.array([1.75, -1.0])
    corner_4 = np.array([-0.85, -1.0])
    
    width = corner_2[0] - corner_1[0]
    height = corner_3[1] - corner_2[1]
    
    center_x = corner_1[0] + width/2 
    center_y = corner_2[1] + height/2
    
    angle_over_traj = []
    speed_over_traj = []
    x_over_traj = []
    y_over_traj = []
    yaw_over_traj = []
    x_vel_over_traj = []
    y_vel_over_traj = []
    yaw_rate_over_traj = [] 
    with mujoco.viewer.launch_passive(m, d) as viewer:

        # Set camera to view from above (top-down view)
        with viewer.lock():
            viewer.cam.distance = 10.0
            viewer.cam.azimuth = 0.0
            viewer.cam.elevation = -20.0
            
        start = time.time()
        control_step = 0
        
        while viewer.is_running() and control_step < steps:
            
            step_start = time.time()
            
            # Get new control command every 20 Hz (every 50ms)
            if control_step % 1 == 0: 
                xy = d.qpos[:2]
                orientation = d.qpos[3:7]
                _, _, yaw = quat2euler(orientation)
                x_vel, y_vel = d.qvel[:2]
                yaw_rate = d.qvel[5]
                obs = np.concatenate([xy, np.array([np.sin(yaw), np.cos(yaw), x_vel, y_vel, yaw_rate])])
                
                action, reference = agent.act(t=control_step, obs=obs)
                angle_over_traj.append(action[0])
                speed_over_traj.append(action[1])
                x_over_traj.append(xy[0])
                y_over_traj.append(xy[1])
                yaw_over_traj.append(yaw)
                x_vel_over_traj.append(x_vel)
                y_vel_over_traj.append(y_vel)
                yaw_rate_over_traj.append(yaw_rate)
            
            # Apply the same action for multiple simulation steps
            d.ctrl[:] = action
            
            sim_steps_per_control = int(control_dt / m.opt.timestep)
            for _ in range(sim_steps_per_control):
                mujoco.mj_step(m, d)
                
                viewer.user_scn.ngeom = 2 
                
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[0],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.02, 0, 0],
                    pos=np.array([reference[0], reference[1], 0.05]),  
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0, 1, 0, 0.8]) 
                )
                
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[1],
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[width/2, height/2, 0.025],  
                    pos=np.array([center_x, center_y, 0.01]),  
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0.5, 0.5, 0.5, 0.25]) 
                )
            
            viewer.sync()

            elapsed = time.time() - step_start
            sleep_time = max(0, control_dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            control_step += 1
            
        end = time.time()
        print(f"Simulation completed in {end - start:.2f} seconds")
        
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(angle_over_traj, linewidth=3, label="Angle")
    axs[1].plot(speed_over_traj, linewidth=3, label="Speed")

    axs[0].axhline(y=np.max(angle_over_traj), color='r', linestyle='--', label="Max Angle")
    axs[0].text(0, np.max(angle_over_traj), f"Max Angle: {np.max(angle_over_traj):.2f}", color='r')
    axs[0].axhline(y=np.min(angle_over_traj), color='r', linestyle='--', label="Min Angle")
    axs[0].text(0, np.min(angle_over_traj), f"Min Angle: {np.min(angle_over_traj):.2f}", color='r')
    axs[1].axhline(y=np.max(speed_over_traj), color='r', linestyle='--', label="Max Speed")
    axs[1].text(0, np.max(speed_over_traj), f"Max Speed: {np.max(speed_over_traj):.2f}", color='r')
    axs[1].axhline(y=np.min(speed_over_traj), color='r', linestyle='--', label="Min Speed")
    axs[1].text(0, np.min(speed_over_traj), f"Min Speed: {np.min(speed_over_traj):.2f}", color='r')
    
    axs[0].legend()
    axs[1].legend()
    axs[0].set_ylabel("Angle (rad)")
    axs[1].set_ylabel("Speed (m/s)")
    axs[0].set_xlabel("Time (s)")
    axs[1].set_xlabel("Time (s)")
    plt.show()
    
    fig, axs = plt.subplots(3, 2)
    axs[0, 0].plot(x_over_traj)
    axs[0, 1].plot(y_over_traj)
    axs[1, 0].plot(yaw_over_traj)
    axs[1, 1].plot(x_vel_over_traj)
    axs[2, 0].plot(y_vel_over_traj)
    axs[2, 1].plot(yaw_rate_over_traj)
    plt.show()
    
    print("Sanity check complete!")

if __name__ == "__main__":
    main()
