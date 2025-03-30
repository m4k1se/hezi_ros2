import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int32
import time
import sys
sys.path.append('/home/renth/mpc_ws/src/demo1/follow_traj_wd/follow_traj_wd')
from follow_demo_2025 import VehicleTrajectoryFollower
from can_use import ISGSpeedFilter
import logging

class FollowNode(Node):
    def __init__(self, main_trajectory_csv, alternate_trajectory_csv):
        super().__init__('Follow_node')
        self.publisher_ = self.create_publisher(
            Float32MultiArray, 
            'planner_action', 
            1
        )
        self.vs_subscription = self.create_subscription(
            Float32MultiArray,
            'vehicle_state',
            self.vs_callback,
            1
        )
        self.eps_subscription = self.create_subscription(
            Int32,
            'eps_mode',
            self.eps_callback,
            1
        )
        self.manual_triggered = True
        self.mode_AE = 1
        self.mode_666 = 1
        self.eps_subscription
        self.vs_subscription  # prevent unused variable warning
        self.follower = VehicleTrajectoryFollower(main_trajectory_csv, 
                                                  alternate_trajectory_csv)
        self.filter = ISGSpeedFilter()
        self.latest_eps_mode = None
        
    def eps_callback(self, msg):
        self.latest_eps_mode = msg.data
        
    def vs_callback(self, msg):
        if self.latest_eps_mode is None:
            self.get_logger().warn("尚未接收到eps_mode，跳过一次控制")
            return
        self.get_logger().info(f"[vs_callback] Received state: {msg.data}")
        self.get_logger().info(f"[vs_callback] EPS mode: {self.latest_eps_mode}")
        eps_mode = self.latest_eps_mode
        start = time.time()
        ego_lat = msg.data[0]
        ego_lon = msg.data[1]
        ego_yaw = msg.data[2]
        ego_v = msg.data[3]
        if eps_mode != 3 and self.manual_triggered:
            self.mode_AE = 1
            self.mode_666 = 1
        if eps_mode ==3:
            self.mode_AE = 3
            self.mode_666 = 0
            self.manual_triggered = False
        if self.mode_AE == 1 and self.mode_666 == 1:
            if ego_lon is not None and ego_lat is not None:
                self.follower.update_closest_indices(ego_lat, ego_lon)
                turn_angle = self.follower.calculate_turn_angle(
                    (ego_lat, ego_lon, ego_yaw), ego_yaw)
                filtered_angle = self.filter.update_speed(turn_angle)
                desired_speed, desired_acc = self.follower.calculate_speedAndacc(
                        turn_angle, (ego_lat, ego_lon, ego_yaw), ego_v, is_obstacle = False)
                logging.info(f'trun angle: {turn_angle}, filter angle: {filtered_angle}')
                self.frame = [float(desired_speed), 
                              float(filtered_angle), 
                              float(desired_acc)]
                planner_frame = Float32MultiArray()
                planner_frame.data = self.frame
                self.get_logger().info(f"[vs_callback] Send frame: {planner_frame.data}")
                self.publisher_.publish(planner_frame)
        elapsed_time = time.time() - start
        self.get_logger().info(f"calc time:{elapsed_time:.6f} [sec]")

def main(args=None):
    main_trajectory_csv = '/home/renth/follow/collect_trajectory/processed_shiyanzhongxin_0327.csv'
    alternate_trajectory_csv = '/home/renth/follow/collect_trajectory/processed_haima-1119-right.csv'
    rclpy.init(args=args)
    follow_node = FollowNode(main_trajectory_csv, alternate_trajectory_csv)
    rclpy.spin(follow_node)
    FollowNode.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()