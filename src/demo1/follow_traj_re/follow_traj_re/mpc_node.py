import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import time
import sys
sys.path.append('/home/renth/mpc_ws/src/demo1/follow_traj_re/follow_traj_re')
from mpc_follower import State, MPCfollower

class MPCNode(Node):

    def __init__(self):
        super().__init__('mpc_node')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'planner_action', 10)
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'vehicle_state',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.mpc = MPCfollower('/home/renth/follow_trajectory/collect_trajectory/processed_straight12_17_with_yaw_ck.csv')

    def listener_callback(self, msg):
        start = time.time()
        start_x = msg.data[0]
        start_y = msg.data[1]
        start_v = msg.data[2]
        start_yaw = msg.data[3]
        initial_state = State(x=start_x, y=start_y, yaw=start_yaw, v=start_v)

        ai, di = self.mpc.cal_acc_and_delta(initial_state)

        elapsed_time = time.time() - start
        self.get_logger().info(f"calc time:{elapsed_time:.6f} [sec]")

        control_msg = Float32MultiArray()
        control_msg.data = [ai, di]  # Assuming d and a are the control actions
        self.publisher_.publish(control_msg)

def main(args=None):
    rclpy.init(args=args)
    mpc_node = MPCNode()
    rclpy.spin(mpc_node)
    mpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()