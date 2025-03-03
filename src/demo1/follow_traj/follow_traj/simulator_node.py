import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import time
import sys
sys.path.append('/home/renth/mpc_ws/src/demo1/follow_traj/follow_traj')
from my import State, read_csv, Simulator

class SimulatorNode(Node):

    def __init__(self):
        super().__init__('simulator_node')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'simulator_state', 10)
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'mpc_control',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.cx, self.cy, self.cyaw, _ = read_csv('/home/renth/follow_trajectory/collect_trajectory/processed_straight12_17_with_yaw_ck.csv')
        start_x = self.cx[0]
        start_y = self.cy[0]
        start_yaw = self.cyaw[0]
        initial_state = State(x=start_x, y=start_y, yaw=start_yaw, v=0.0)
        print(f"Initial state: {initial_state.x, initial_state.y, initial_state.yaw}")
        self.sim = Simulator(initial_state)

        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        pose_msg = Float32MultiArray()
        pose_msg.data = [self.sim.state.x, self.sim.state.y, self.sim.state.v, self.sim.state.yaw]
        self.publisher_.publish(pose_msg)

    def listener_callback(self, msg):
        ai = msg.data[0]
        di = msg.data[1]
        # self.sim.update_state(ai, di)
        self.sim.do_simulation(ai, di, self.cx, self.cy, self.cyaw, 0)

def main(args=None):
    rclpy.init(args=args)
    simulator_node = SimulatorNode()
    rclpy.spin(simulator_node)
    simulator_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()