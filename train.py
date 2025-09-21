from sym2real.util.replay_buffer import ReplayBuffer
import os
import pandas as pd
import numpy as np
import hydra
import omegaconf

from sym2real.util.common import (
    create_one_dim_tr_model,
    train_model_and_save_model_and_data,
)
from sym2real.dynamics_model.model_trainer import JAXModelTrainer
from sym2real.util.logger import Logger
from utils.model_eval import evaluate_model

# Custom Hydra resolver for clean env_params strings
def clean_env_params(env_params):
    return "_".join(f"{k}_{v}" for k, v in env_params.items())

# Custom Hydra resolver for dynamics model directory names
def get_dynamics_model_dir(dynamics_model_choice, dynamics_model_config):
    if dynamics_model_choice.startswith("residual"):
        try:
            if isinstance(dynamics_model_config, str):
                if 'model_a' in dynamics_model_config:
                    import re
                    match = re.search(r"'model_a':\s*'([^']*)'", dynamics_model_config)
                    if match:
                        model_a_value = match.group(1)
                        return f"{model_a_value}_{dynamics_model_choice}"
            else:
                if hasattr(dynamics_model_config, 'model_a') and dynamics_model_config.model_a:
                    return f"{model_a_value}_{dynamics_model_choice}"
        except:
            pass
        return dynamics_model_choice
    else:
        return dynamics_model_choice

# Register the custom resolvers
omegaconf.OmegaConf.register_new_resolver("clean_env_params", clean_env_params)
omegaconf.OmegaConf.register_new_resolver("get_dynamics_model_dir", get_dynamics_model_dir)

def create_replay_buffer_from_real_data(cfg):
    if "Mushr" in cfg.overrides.env:
        replay_buffer = ReplayBuffer(capacity=int(1e6),
                                obs_shape=(7,),
                                action_shape=(2,),
        )
    else:
        assert False, "This env is not supported yet"
    
    work_dir = os.getcwd()
    print(f"Current working directory: {work_dir}")
    csv_folders = os.listdir(os.path.join(work_dir, "real_data"))
    csv_full_folders = [os.path.join(work_dir, "real_data", each_folder) for each_folder in csv_folders]
    
    for each_folder in csv_full_folders:
        csv_files = [f for f in os.listdir(each_folder) if f.endswith('.csv')]
        
        for each_file in csv_files:
            if each_file == "traj_1.csv":
                file_path = os.path.join(each_folder, each_file)
                data = pd.read_csv(file_path)
                
                if "Mushr" in cfg.overrides.env:
                    # MuSHR data format: [time, x, y, yaw, x_vel, y_vel, yaw_rate, cmd_angle, cmd_speed, x_ref, y_ref, yaw_ref, x_vel_ref, y_vel_ref, yaw_rate_ref]
                    # Extract states: [x, y, yaw, x_vel, y_vel, yaw_rate]
                    x, y, yaw, x_vel, y_vel, yaw_rate = data["x"].values, data["y"].values, data["yaw"].values, data["x_vel"].values, data["y_vel"].values, data["yaw_rate"].values
                    
                    # Convert to observation format: [x, y, sin_yaw, cos_yaw, x_vel, y_vel, yaw_rate]
                    states_temp = np.stack([x, y, np.sin(yaw), np.cos(yaw), x_vel, y_vel, yaw_rate], axis=1)
                    
                    # Extract actions: [cmd_angle, cmd_speed]
                    actions_temp = data[["cmd_angle", "cmd_speed"]].values
                    
                    # Create next states (shift by 1)
                    obs = states_temp[:-1]  # Exclude last row since it has no next state
                    actions = actions_temp[:-1]  # Align with states
                    next_obs = states_temp[1:]  # Next states
                else:
                    assert False, "This env is not supported yet"
                
                rewards = np.zeros(len(obs)) + np.random.normal(0, 0.01, len(obs))  # dummy rewards
                truncated = np.zeros(len(obs), dtype=bool)
                terminated = np.zeros(len(obs), dtype=bool)
                
                for i in range(len(obs)):
                    replay_buffer.add(
                        obs[i], 
                        actions[i], 
                        next_obs[i],
                        rewards[i], 
                        bool(terminated[i]), 
                        bool(truncated[i]), 
                    )
            
    
    replay_buffer.save(work_dir)
    
    print(f"Replay buffer created with {replay_buffer.num_stored} samples.")
    
    return replay_buffer 

if __name__ == "__main__":

    @hydra.main(config_path="../sym2real/conf", config_name="main_real_mushr", version_base="1.1")
    def run(cfg: omegaconf.DictConfig):
        work_dir = os.getcwd()
        logger = Logger(work_dir)
        """
        First create the replay buffer from real data.
        """
        
        replay_buffer = create_replay_buffer_from_real_data(cfg)
                
        train_replay_buffer = ReplayBuffer(capacity=int(1e6),
                                obs_shape=(7,),
                                action_shape=(2,),
        )
        intial_samples = replay_buffer.num_stored; 
        
        # split to train and test
        train_replay_buffer.obs = replay_buffer.obs[0:intial_samples] 
        train_replay_buffer.action = replay_buffer.action[0:intial_samples] 
        train_replay_buffer.next_obs = replay_buffer.next_obs[0:intial_samples] 
        train_replay_buffer.reward = replay_buffer.reward[0:intial_samples] 
        train_replay_buffer.terminated = replay_buffer.terminated[0:intial_samples] 
        train_replay_buffer.truncated = replay_buffer.truncated[0:intial_samples] 
        train_replay_buffer.num_stored = intial_samples - 0
        """
        Then create the model
        """
        
        # ╔════════════════════════════════════════════════════════════════════╗
        # ║         Create a 1-D dynamics model for this environment           ║
        # ╚════════════════════════════════════════════════════════════════════╝

        if "Mushr" in cfg.overrides.env:
            obs_shape = (7,)
            act_shape = (2,)
            
        if cfg.model_path == "":
            dynamics_model = create_one_dim_tr_model(cfg=cfg,
                                                obs_shape=obs_shape, 
                                                act_shape=act_shape,
                                                )
        else:
            dynamics_model = create_one_dim_tr_model(cfg=cfg, 
                                                obs_shape=obs_shape, 
                                                act_shape=act_shape,
                                                model_dir=cfg.model_path,
                                                )

        model_trainer = JAXModelTrainer(model=dynamics_model,
                            logger=logger)
            
        if "MLP" in cfg.dynamics_model._target_:
            optimizer = model_trainer.create_optimizer(model=dynamics_model.model,
                                                        learning_rate=cfg.overrides.get("model_lr", 1e-3),
                                                        weight_decay=cfg.overrides.get("model_wd", 1e-5),
                                                        eps=cfg.overrides.get("optim_eps", 1e-8))
        elif ("Residual" in cfg.dynamics_model._target_ and "MLP" in cfg.dynamics_model.model_b._target_):
            optimizer = model_trainer.create_optimizer(model=dynamics_model.model.model_B,
                                                        learning_rate=cfg.overrides.get("model_lr", 1e-3),
                                                        weight_decay=cfg.overrides.get("model_wd", 1e-5),
                                                        eps=cfg.overrides.get("optim_eps", 1e-8))
        else:
            optimizer = None

        # """
        # Then train the residual model
        # """
        save_dir = train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg,
                    optimizer,
                    train_replay_buffer,
                    work_dir=work_dir,
                    env_steps=intial_samples,
                )
        
        # --------------- Model evaluation (optional) ---------------
        range_start = np.array([0, 25, 50, 100, 200, 300, 350])
        for each_range in range_start:
            if each_range + 40 <= replay_buffer.num_stored:  # Check bounds
                evaluate_model(
                    dynamics_model,
                    replay_buffer,
                    indices=np.arange(int(each_range), int(each_range)+40),
                    save_dir=os.path.join(work_dir, "{each_range}_model_eval_figs".format(each_range=each_range)),
                    type="mlp"
                )
        
    run()
