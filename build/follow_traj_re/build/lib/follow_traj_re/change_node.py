import math
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String
from lane_change import LaneChangeDecider
from can_use import Can_use
from mpc_follower import State

def pi_2_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def read_csv(csv_file_path): 
    x_coords = []
    y_coords = []
    heading_list = []
    speed_list = []
    with open(csv_file_path, mode='r', newline='') as file:  
        csv_reader = csv.reader(file)  
        headers = next(csv_reader, None)  # 跳过标题行
        for row in csv_reader:  
            lon = float(row[0])  
            lat = float(row[1])
            heading = float(row[2])
            x_coords.append(lon)
            y_coords.append(lat)
            heading_list.append(heading)
            speed_list.append(float(row[3]))
    return x_coords, y_coords, heading_list, speed_list

class ChangeNode(Node):
    def __init__(self, path):
        super().__init__('Change_node')
        self.decider = LaneChangeDecider(path)
        self.can_use = Can_use(zone=49)
        self.target_speed = 10.0 / 3.6
        self.cx, self.cy, self.cyaw, self.ck = read_csv(path)
        self.sp = self.decider.calc_speed_profile(self.target_speed)
        self.cyaw = self.decider.smooth_yaw(self.cyaw)
        self.obs_list = []
        self.all_line_temp = []
        self.ref_line = []
        self.new_line = []
        self.state = None
        self.decider.init_refline()
        self.planning = False
        # ROS 2 Publisher
        self.publisher_ = self.create_publisher(Float32MultiArray, 'new_refline', 1)
        self.timer_ = self.create_timer(0.01, self.publish_new_refline)
        self.obs_subscription = self.create_subscription(
            String,
            'image_detection_results',
            self.listener_callback,
            1
        )
    
    def listener_callback(self, msg):
        data = msg.data
        print("接收到检测结果: ", data)
        self.get_logger().info("接收到检测结果: " + data)
        try:
            # 按照'|'分割，获取目标总数和坐标信息
            header, coordinates_str = data.split('|', 1)
            header = header.strip()  # 例如："Total targets: 3"
            total_targets = int(header.split(':')[1].strip())
            self.get_logger().info("检测到目标总数: %d" % total_targets)

            coordinates_list = []
            coordinates_str = coordinates_str.strip()
            if coordinates_str:
                # 以分号分割每个目标的坐标字符串
                targets = coordinates_str.split(';')
                for target in targets:
                    target = target.strip()
                    if target:
                        x_str, y_str, z_str, pixel_cx, pixel_cy, pixel_w, pixel_h = target.split(',')
                        x        = float(x_str)  # 真实的距离
                        y        = float(y_str)  # 
                        z        = float(z_str)
                        pixel_cx = float(pixel_cx) # 图像坐标系的
                        pixel_cy = float(pixel_cy)
                        pixel_w  = float(pixel_w)
                        pixel_h  = float(pixel_h)
                        coordinates_list.append([x, y, z, pixel_cx, pixel_cy, pixel_w, pixel_h])
            # 输出各目标的坐标
            for idx, (x, y, z, pixel_cx, pixel_cy, pixel_w, pixel_h) in enumerate(coordinates_list, start=1):
                self.get_logger().info("目标 %d: x=%.2f, y=%.2f, z=%.2f" % (idx, x, y, z))
        except Exception as e:
            self.get_logger().error("解析检测结果失败: " + str(e))
        self.obs_list = coordinates_list

    def publish_new_refline(self):
        for i in range(30):
            self.can_use.read_ins_info()
        x = self.can_use.ego_x
        y = self.can_use.ego_y
        v = self.can_use.ego_v
        yaw = self.can_use.ego_yaw
        initial_state = State(x=x, y=y, yaw=yaw, v=v)
        print("obs_list: ", self.obs_list)
        self.decider.update_state(initial_state, self.obs_list)
        self.cx, self.cy, self.cyaw, self.ck, self.sp = self.decider.publish_new_refline()
        
        # 创建 Float32MultiArray 消息
        msg = Float32MultiArray()
        
        # 将 cx, cy, cyaw, ck 和 sp 数据打包
        data = self.cx + self.cy + self.cyaw + self.ck + self.sp
        
        # 设置 layout 信息
        msg.layout.dim.append(MultiArrayDimension(label='cx', size=len(self.cx), stride=len(data)))
        msg.layout.dim.append(MultiArrayDimension(label='cy', size=len(self.cy), stride=len(self.cy) + len(self.cx)))
        msg.layout.dim.append(MultiArrayDimension(label='cyaw', size=len(self.cyaw), stride=len(self.cyaw) + len(self.cy) + len(self.cx)))
        msg.layout.dim.append(MultiArrayDimension(label='ck', size=len(self.ck), stride=len(self.ck) + len(self.cyaw) + len(self.cy) + len(self.cx)))
        msg.layout.dim.append(MultiArrayDimension(label='sp', size=len(self.sp), stride=len(data)))
        
        # 设置数据
        msg.data = data
        
        # 发布消息
        self.publisher_.publish(msg)
        self.get_logger().info("Published new refline data with sp")

def main(args=None):
    rclpy.init(args=args)
    lane_change_decider = ChangeNode('/home/renth/follow/collect_trajectory/processed_straight12_17_with_yaw_ck.csv')
    rclpy.spin(lane_change_decider)
    lane_change_decider.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()