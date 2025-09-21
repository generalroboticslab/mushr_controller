import rospy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped

def send_command(pub_controls, delta: float, v: float, duration: float):
    dur = rospy.Duration(duration)
    rate = rospy.Rate(10)
    start = rospy.Time.now()

    drive = AckermannDrive(steering_angle=delta, speed=v)

    while rospy.Time.now() - start < dur and not rospy.is_shutdown():
        msg = AckermannDriveStamped()
        msg.drive = drive
        pub_controls.publish(msg)
        rate.sleep()

if __name__ == "__main__":
    rospy.init_node("path_publisher")

    control_topic = rospy.get_param(
        "~control_topic",
        "/car/mux/ackermann_cmd_mux/input/navigation"
    )
    pub_controls = rospy.Publisher(control_topic, AckermannDriveStamped, queue_size=10)

    rospy.loginfo("Warming up publisher...")
    rospy.sleep(1.0)   # let ROS establish connections

    # Example usage
    delta = 0.1  # Steering angle in radians
    v = 0.1      # Speed in m/s
    rospy.loginfo("Sending command: delta=%.2f, v=%.2f", delta, v)

    send_command(pub_controls, delta, v, duration=3)
