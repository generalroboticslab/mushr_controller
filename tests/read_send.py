import rospy
import csv
from geometry_msgs.msg import TransformStamped
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from utils.math import quat2euler
import os
import numpy as np

class ViconMPCNode:
    def __init__(self, log_dir="data_logs"):
        rospy.init_node("vicon_mpc_node")

        # Subscribers & Publishers
        self.pose_sub = rospy.Subscriber("/vicon/mushr/mushr", TransformStamped, self.read_pose)
        self.control_pub = rospy.Publisher(
            rospy.get_param("~control_topic", "/car/mux/ackermann_cmd_mux/input/navigation"),
            AckermannDriveStamped,
            queue_size=10
        )
        
        # Control parameters
        self.current_pose = None
        self.rate = rospy.Rate(20)  # 20 Hz control loop
        self.drive_msg = AckermannDriveStamped()
        
        # Logging setup
        os.makedirs("data_logs", exist_ok=True)
        log_path = os.path.join(log_dir, "state_action_log_5.csv")
        self.log_file = open(log_path, "w", newline="")
        self.logger = csv.writer(self.log_file)
        self.logger.writerow(["time", "x", "y", "yaw", "cmd_angle", "cmd_speed"])

    def get_action(self):
        """Placeholder for MPC control logic (to be implemented later)."""
        # get random delta
        # get random v
        
        delta = np.random.uniform(-0.2, 0.2)
        v = np.random.uniform(-0.5, 0.5)
        return delta, v

    def read_pose(self, msg):
        """Callback to read Vicon pose data."""
        x, y = msg.transform.translation.x, msg.transform.translation.y
        qx, qy, qz, qw = msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w
        _, _, yaw = quat2euler(qx, qy, qz, qw)
        self.current_pose = (x, y, yaw)

    def send_control(self, delta, v):
        """Sends control commands while logging state-action pairs."""
        if self.current_pose is None:
            rospy.logwarn("No Vicon data received yet.")
            return
        
        self.drive_msg.drive = AckermannDrive(steering_angle=delta, speed=v)
        self.control_pub.publish(self.drive_msg)
        self.log_state_action(delta, v)

    def log_state_action(self, delta, v):
        """Logs current state and action."""
        if self.current_pose is None:
            return

        timestamp = rospy.Time.now().to_sec()
        x, y, yaw = self.current_pose
        self.logger.writerow([timestamp, x, y, yaw, delta, v])

    def run(self):
        """Main control loop that reads Vicon data, gets action, sends control, and logs data."""
        rospy.loginfo("Waiting for Vicon data...")
        while self.current_pose is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

        rospy.loginfo("Starting control loop...")
        try:
            while not rospy.is_shutdown():
                delta, v = self.get_action()
                self.send_control(delta, v)

                # Maintain loop frequency
                self.rate.sleep()
        except Exception as e:
            rospy.logerr(f"Error occurred: {e}")
        finally:
            self.log_file.close()
            rospy.loginfo("Log file closed. Node shutting down.")

if __name__ == "__main__":
    node = ViconMPCNode(log_dir="/home/generalroboticslab/Desktop//mushr_symbolic/data_logs")
    node.run()
