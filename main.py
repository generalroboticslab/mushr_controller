import rospy
import csv
import os
import numpy as np
import omegaconf
import signal
import time
import sys

from geometry_msgs.msg import TransformStamped
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

from sym2real.util.common import create_one_dim_tr_model
from sym2real.controllers.agent import TrajectoryPlanningAgent
from sym2real.controllers.reference_trajectories.setpoint_ref import SetpointReference
from sym2real.controllers.reference_trajectories.mushr_circle_ref import MushrCircleReference
from sym2real.controllers.reference_trajectories.mushr_oval_ref import MushrTrackReference
from sym2real.controllers.mpc import CEMController, SplineMPPIController
from utils.math import quat2euler

import sym2real.envs
import gymnasium as gym
import hydra
from collections import deque

# WORKSPACE (same as main.py)
corner_1 = np.array([-1.2, -4])
corner_2 = np.array([2.40, -4])
corner_3 = np.array([2.40, 2.0])
corner_4 = np.array([-1.2, 2.0])

MIN_X = corner_1[0] - 0.2 # extra 0.1m in each direction
MAX_X = corner_2[0] + 0.2
MIN_Y = corner_2[1] - 0.2
MAX_Y = corner_3[1] + 0.2 # extra 0.1m in each direction

TRAJ_TIME_IN_SEC = 20.0
CTRL_HZ = 20.0

is_in_workspace = lambda x, y: MIN_X <= x <= MAX_X and MIN_Y <= y <= MAX_Y

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully to avoid deadlocks."""
    print("\nReceived interrupt signal. Shutting down gracefully...")
    
    try:
        import jax
        jax.clear_caches()
        jax.device_get(jax.numpy.array([1.0]))
        print("✓ JAX GPU resources cleaned up")
    except Exception as e:
        print(f"⚠ JAX cleanup warning: {e}")
    
    try:
        import pysr
        import gc
        gc.collect()
        print("✓ Julia resources cleaned up")
    except Exception as e:
        print(f"⚠ Julia cleanup warning: {e}")
    
    print("Cleanup complete. Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class MuSHRControlNode:
    def __init__(self, cfg, sim_env, which_trial, traj_num, enable_logging=True):
        self.cfg = cfg
        self.sim_env = sim_env
        self.enable_logging = enable_logging
        self.pose_buffer = deque(maxlen=5)  # median filter window
        
        self.max = 0
        work_dir = os.getcwd()

        if self.enable_logging:
            folder = f"{work_dir}/real_data/"
            if not os.path.exists(folder):
                os.makedirs(folder)
                
            trial_folder = f"{folder}/trial_{which_trial}"
            if not os.path.exists(trial_folder):
                os.makedirs(trial_folder)
            filename = f"{trial_folder}/traj_{traj_num}.csv"
            
            # Create CSV logger similar to main.py
            self.log_file = open(filename, "w", newline="")
            self.logger = csv.writer(self.log_file)
            self.logger.writerow(["time", "x", "y", "yaw", "x_vel", "y_vel", "yaw_rate", "cmd_angle", "cmd_speed", "x_ref", "y_ref", "yaw_ref", "x_vel_ref", "y_vel_ref", "yaw_rate_ref"])
            
        # ----------------------------------------------------------------------
        #  DYNAMICS MODEL
        # ----------------------------------------------------------------------
        self.dynamics_model = create_one_dim_tr_model(cfg=cfg, 
                                                      obs_shape=sim_env.observation_space.shape, 
                                                      act_shape=sim_env.action_space.shape,
                                                      model_dir=cfg.model_path
                                                      )
        
        # ----------------------------------------------------------------------
        #  CONTROL AGENT (MPPI)
        # ----------------------------------------------------------------------
        if cfg.overrides.reference_type == "circle":
            self.reference_class = MushrCircleReference(radius=1.5, backwards=False)
        elif cfg.overrides.reference_type == "hover":
            self.reference_class = SetpointReference()
        elif cfg.overrides.reference_type == "track":
            self.reference_class = MushrTrackReference()
        else:
            # Default to circle reference
            self.reference_class = MushrCircleReference(radius=1.5, backwards=False)
        
        rollout_horizon_in_sec = 1.0
        num_samples = 1024
        
        self.controller = SplineMPPIController(
            env=sim_env,
            horizon=int(rollout_horizon_in_sec/sim_env.unwrapped.dt),
            dt=sim_env.unwrapped.dt,
            num_samples=num_samples,
            model=self.dynamics_model,
            seed=cfg.seed
        )

        self.agent = TrajectoryPlanningAgent(self.controller, reference_class=self.reference_class)
        
        # Initialize ROS node
        rospy.init_node("mushr_vicon_ctrl", anonymous=True)
        self.rate = rospy.Rate(CTRL_HZ)
        
        # Subscribers & Publishers
        self.pose_sub = rospy.Subscriber("/vicon/mushr/mushr", TransformStamped, self.read_pose)
        self.control_pub = rospy.Publisher(
            rospy.get_param("~control_topic", "/car/mux/ackermann_cmd_mux/input/navigation"),
            AckermannDriveStamped,
        )

        # Control parameters
        self.current_pose = None
        self.prev_pose = None
        self.prev_time = None
        self.drive_msg = AckermannDriveStamped()

        # Wait for initial pose
        rospy.loginfo("Waiting for Vicon data...")
        while self.current_pose is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

        # Set reference center based on initial position
        x_init, y_init, yaw_init, _, _, _ = self.current_pose
        print(f"x_init: {x_init}, y_init: {y_init}, yaw_init: {yaw_init}")
        
        if isinstance(self.reference_class, MushrCircleReference):
            self.reference_class.set_center([x_init, y_init, 1.0], initial_yaw=yaw_init)
        elif isinstance(self.reference_class, SetpointReference):
            self.reference_class.set_center([x_init, y_init, yaw_init])
        elif isinstance(self.reference_class, MushrTrackReference):
            self.reference_class.set_center([x_init, y_init, 1.0])
        
        self.step_ct = 0
        
        # Warm-up MPPI model
        for i in range(3):
            self.agent.act(t=0, obs=np.zeros((7,)))
        
        print("MuSHR Control Node initialized successfully!")
    
    def read_pose(self, msg):
        """Callback to read Vicon pose data and compute velocity."""
        x, y = msg.transform.translation.x, msg.transform.translation.y
        qx, qy, qz, qw = msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w
        _, _, yaw = quat2euler(qx, qy, qz, qw)

        curr_time = msg.header.stamp.to_sec()

        if self.prev_pose is None:
            self.prev_pose = (x, y, yaw)
            self.prev_time = curr_time
            self.current_pose = (x, y, yaw, 0.0, 0.0, 0.0)
            return
        
        def angle_diff(a, b):
            """Return smallest signed difference between two angles (radians)."""
            d = a - b
            return (d + np.pi) % (2 * np.pi) - np.pi
        
        # --- median filter on raw poses ---
        self.pose_buffer.append((x, y, yaw))
        x_m = np.median([p[0] for p in self.pose_buffer])
        y_m = np.median([p[1] for p in self.pose_buffer])
        yaw_m = np.median([p[2] for p in self.pose_buffer])
        x, y, yaw = x_m, y_m, yaw_m

        dt = curr_time - self.prev_time
        if dt > 0:
            x_vel = (x - self.prev_pose[0]) / dt
            y_vel = (y - self.prev_pose[1]) / dt
            yaw_rate = angle_diff(yaw, self.prev_pose[2]) / dt
            
            # --- outlier rejection ---
            max_speed, max_yaw_rate = 5.0, 5.0  # tune to your car
            if abs(x_vel) > max_speed or abs(y_vel) > max_speed or abs(yaw_rate) > max_yaw_rate:
                # Ignore outlier frame
                return

            self.current_pose = (x, y, yaw, x_vel, y_vel, yaw_rate)

        self.prev_pose = self.current_pose
        self.prev_time = curr_time

    def send_control(self, delta, v):
        """Sends control commands while logging state-action pairs."""
        if self.current_pose is None:
            rospy.logwarn("No Vicon data received yet.")
            return

        self.drive_msg.drive = AckermannDrive(steering_angle=delta, speed=v)
        self.control_pub.publish(self.drive_msg)

    def log_state_action(self, delta, v, reference):
        """Logs current state and action."""
        if self.current_pose is None:
            return

        timestamp = rospy.Time.now().to_sec()
        x, y, yaw, x_vel, y_vel, yaw_rate = self.current_pose
        x_ref, y_ref, yaw_ref, x_vel_ref, y_vel_ref, yaw_rate_ref = reference
        self.logger.writerow([timestamp, x, y, yaw, x_vel, y_vel, yaw_rate, delta, v, x_ref, y_ref, yaw_ref, x_vel_ref, y_vel_ref, yaw_rate_ref])

    def run(self):
        """Main control loop using Vicon data with MPPI controller."""
        rospy.loginfo("Starting control loop...")
        
        try:
            t = 0
            while not rospy.is_shutdown() and t <= int(TRAJ_TIME_IN_SEC/0.05):

                if t > int(TRAJ_TIME_IN_SEC/0.05):
                    print(f"Loop exited: Time limit reached ({t} > {TRAJ_TIME_IN_SEC})")
                    break
                
                x, y, yaw, x_vel, y_vel, yaw_rate = self.current_pose
                
                if not is_in_workspace(x, y):
                    print(f"Loop exited: Car left workspace at t={t}, pos=({x:.3f}, {y:.3f})")
                    print(f"Workspace bounds: X[{MIN_X:.3f}, {MAX_X:.3f}], Y[{MIN_Y:.3f}, {MAX_Y:.3f}]")
                    break
                
                sin_yaw = np.sin(yaw)
                cos_yaw = np.cos(yaw)
                obs = np.array([x, y, sin_yaw, cos_yaw, x_vel, y_vel, yaw_rate])
                print(f"Current state: x: {x}, y: {y}, yaw: {yaw}, x_vel: {x_vel}, y_vel: {y_vel}, yaw_rate: {yaw_rate}")
                
                # Get action from MPPI controller
                start_act = time.time()
                action, reference = self.agent.act(t, obs)
                print(f"Reference: {reference}")
                end_act = time.time()
                
                self.max = max(self.max, end_act - start_act)
                print(f"MPPI action took {end_act - start_act:.3f} seconds.")
                
                delta, v = action
                # Apply steering bias
                post_processed_delta = delta + 0.1

                print(f"delta: {delta}, v: {v}")
                self.send_control(post_processed_delta, v)
                
                if self.enable_logging:
                    self.log_state_action(delta, v, reference)

                t += 1
                self.rate.sleep()
                
        except Exception as e:
            rospy.logerr(f"Error occurred: {e}")
        finally:
            delta, v = 0, 0
            self.send_control(delta, v)
            self.rate.sleep()
            
            if self.enable_logging:
                self.log_file.close()
                rospy.loginfo("Log file closed.")
            
            rospy.loginfo("Node shutting down.")
            print("Max MPPI time: ", self.max)
    
if __name__ == "__main__":
    @hydra.main(config_path="../sym2real/conf", config_name="main_real_mushr", version_base="1.1")
    def run(cfg: omegaconf.DictConfig):
        
        safety_check = input("Do you want to create a new set of experiments? Then quit and fix the config file. [y/n]: ").strip().lower()
        if safety_check == "y":
            print("Quitting... Please change experiment name in the config file.")
            return
        elif safety_check == "n":
            which_trial = input("Please enter the trial number: ")
            try:
                which_trial = int(which_trial)
            except ValueError:
                print("Invalid input. Please enter a number.")
                return
        
        while True:
            cont = input("Collect another trajectory? [y/n]: ").strip().lower()
            if cont != "y":
                break
            
            traj_num = int(input("Enter traj number (reset car before entering this): "))
            
            if cfg.overrides.env == "MushrCar-v0":
                pass 
            else:
                raise ValueError("Invalid environment name. Please use 'MushrCar-v0'.")

            sim_env = gym.make(cfg.overrides.env, 
                            cfg=cfg,
                        render_mode="human",
                        )
            
            node = MuSHRControlNode(cfg,
                                    sim_env,
                                    which_trial,
                                    traj_num,
                                    enable_logging=True)
            node.run()

    run()