import subprocess
import matplotlib.pyplot as plt

cmd = ["rostopic", "echo", "-p", "/vicon/mushr/mushr"]

timestamps = []

with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) as proc:
    for line in proc.stdout:
        if line.startswith('%'):
            continue  # Skip CSV header
        try:
            ts = float(line.strip().split(',')[0])
            timestamps.append(ts)
        except:
            continue

        if len(timestamps) >= 201:  
            break

# Compute deltas
deltas = [(t2 - t1)/1e9 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]

# Plot
plt.figure(figsize=(10, 4))
plt.plot(deltas, marker='o', color="red")
plt.axhline(0.01, color='gray', linestyle='--', label='Expected 100 Hz (10 ms)')
plt.xlabel("Frame Index")
plt.ylabel("Delta Time (s)")
plt.ylim(-0.003, 0.03)
plt.title("Vicon Frame Intervals")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# plt.savefig("vicon_frame_intervals_w_ethernet.png")
