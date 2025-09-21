import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib import transforms
from sym2real.controllers.reference_trajectories.mushr_circle_ref import MushrCircleReference

def generate_video_from_real_traj(log_file, 
                                  goal_position=[1,1], 
                                  record_video=False, 
                                  video_name="output.mp4"):
    
    reference = MushrCircleReference()
    

    dat = pd.read_csv(log_file)
    x = dat["x"].to_numpy()
    y = dat["y"].to_numpy()
    x_ref = dat["x_ref"].to_numpy()
    y_ref = dat["y_ref"].to_numpy()

    reference.set_center([x[0], y[0], 0])
    ref_points = []
    for i in range(400):
        ref_points.append(np.array([x_ref[i], y_ref[i]]))
    yaw = dat["yaw"].to_numpy()
    t = dat["time"].to_numpy()
    
    # Compute time intervals (convert to milliseconds)
    dt = np.diff(t, prepend=t[0]) * 1000 
    fps = 1000 / dt.mean() 

    # Calculate bounds from actual data to encompass everything
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    # Add padding around the data
    padding = 0.3
    xlim = (x_min - padding, x_max + padding)
    ylim = (y_min - padding, y_max + padding)
    
    car_radius = 0.1 
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # To match the video recording orientation
    base = ax.transData
    transform = transforms.Affine2D().rotate_deg(90) + base
    ax.set_transform(transform)

    ax.set_xlabel("y", fontsize=12)
    ax.set_ylabel("x", fontsize=12)

    y_ticks = np.linspace(ylim[0], ylim[1], num=5) 
    ax.set_xticks(y_ticks)
    ax.set_xticklabels([f"{-ytick:.1f}" for ytick in y_ticks])  
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{ytick:.1f}" for ytick in y_ticks])  

    ax.set_facecolor("#d9e6f2")  
    
    rect = plt.Rectangle((xlim[0], ylim[0]), xlim[1] - xlim[0], ylim[1] - ylim[0], 
                         fill=False, edgecolor='#555555', linewidth=1.5, transform=transform)
    ax.add_patch(rect)

    initial_circle = plt.Circle((x[0], y[0]), car_radius, color='white', fill=False, lw=2, transform=transform)
    ax.add_patch(initial_circle)
    arrow_length = car_radius * 1.2
    initial_arrow = ax.arrow(x[0], y[0], arrow_length * np.cos(yaw[0]), arrow_length * np.sin(yaw[0]), 
                             head_width=0.07, head_length=0.07, fc='black', ec='black', lw=1.5, transform=transform)

    car_circle = plt.Circle((x[0], y[0]), car_radius, color='#00a8cc', fill=True, transform=transform)  
    ax.add_patch(car_circle)
    
    # Add moving reference point circle
    ref_circle = plt.Circle((x_ref[0], y_ref[0]), 0.08, color='#ff6b35', fill=True, alpha=0.8, transform=transform)
    ax.add_patch(ref_circle)
    
    # Plot all reference points as small dots
    for point in ref_points:
        ax.plot(point[0], point[1], 'o', color='black', markersize=2, transform=transform)

    trace, = ax.plot([], [], color='#00a8cc', lw=1.5, alpha=0.8, label="Trajectory", transform=transform)
    
    arrow = None

    def update(frame):
        nonlocal arrow 
        
        car_circle.set_center((x[frame], y[frame]))
        
        # Update reference circle position
        ref_circle.set_center((x_ref[frame], y_ref[frame]))
        
        end_x = x[frame] + car_radius * np.cos(yaw[frame])
        end_y = y[frame] + car_radius * np.sin(yaw[frame])
        
        if arrow:
            arrow.remove()
        arrow = ax.arrow(x[frame], y[frame], end_x - x[frame], end_y - y[frame], 
                         head_width=0.07, head_length=0.07, fc='#ffffff', ec='black', lw=1.5, transform=transform)

        trace.set_data(x[:frame+1], y[:frame+1])  # Update trajectory
        return car_circle, ref_circle, arrow, trace, initial_circle, initial_arrow

    legend_elements = [
        plt.Line2D([0], [0], color='#00a8cc', lw=2, label="Trajectory"),  # Trajectory
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff6b35', markersize=8, label="Reference"),  # Reference point
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=8, markeredgecolor='black', label="Start"),  # Start position
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.35, 1))  

    ani = animation.FuncAnimation(fig, update, frames=len(x), interval=dt.mean(), blit=False)

    if record_video:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        ani.save(video_name, writer=writer)
        print("Video saved successfully.")
    else:
        plt.show()

if __name__ == "__main__":
    # folder_name = "/home/generalroboticslab/Desktop/mushr_symbolic/real_exp_results/offset_0.1_sr_mlp/circle/seed=1/real_data/trial_1"
    folder_name = "/home/generalroboticslab/Desktop/mushr_symbolic/real_exp_results/plain_mlp_mlp/circle/seed=1/real_data/trial_1"
    for i in range(1,4):
        generate_video_from_real_traj(
            f"{folder_name}/traj_{i}.csv",
            goal_position=[1,0],
            record_video=True, 
            video_name=f"{folder_name}/traj_{i}.mp4"
        )
