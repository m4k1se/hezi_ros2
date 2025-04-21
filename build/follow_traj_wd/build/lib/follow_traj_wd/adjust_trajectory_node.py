#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import String  # 障碍物信息：这里只是举例
import math
import csv
import sys
import re


import csv  
from pyproj import Proj
import matplotlib.pyplot as plt

sys.path.append('/home/nvidia/vcii/hezi_ros2/src/demo1/follow_traj_wd/follow_traj_wd')
from read_csv import read_mpc_csv
from lane_change_vx import LaneChangeDecider

from switch_trajectory import AdjustTrajectory, obstacle_position
from can_use import Can_use
from utils import Bbox, State

lonlat2xy_old = Proj('+proj=tmerc +lon_0=118.8170043 +lat_0=31.8926311 +ellps=WGS84')
def read_csv(csv_file_path):
    traj_data = []
    x = []
    y = []
    # 打开CSV文件并读取内容  
    with open(csv_file_path, mode='r', newline='') as file:  
        csv_reader = csv.reader(file)  
        
        # 跳过标题行（如果有的话）  
        headers = next(csv_reader, None)  # 这行代码会读取第一行，如果第一行是标题则跳过  
        
        # 读取每一行数据并添加到列表中  
        for row in csv_reader:  
            # 将每一行的数据转换为整数或浮点数（根据具体情况选择）  
            # 假设x坐标、y坐标、航向角和速度都是浮点数  
            x_coord = float(row[0])  
            y_coord = float(row[1])
            x_coord_utm, y_coord_utm = lonlat2xy_old(x_coord, y_coord, inverse=False)  
            heading = float(row[2])  
            x.append(x_coord_utm)
            y.append(y_coord_utm)
            # 将这些信息存储为一个列表，并添加到data_list中  
            data_row = [x_coord, y_coord, heading]  
            traj_data.append(data_row)
    return traj_data  

# TODO:
def adjust_trajectory(traj_data, obstacle_msg):
    """
    轨迹修正函数示例。
    obstacle_msg: 订阅到的障碍物信息(这里String类型)。
    通过解析其中的坐标/大小/速度等信息来计算避障后的轨迹。
    
    这里仅演示：在原始 traj_data 基础上对 x 偏移 +0.1
    """
    new_traj_data = []
    for point in traj_data:
        # point: [x_utm, y_utm, heading]
        new_point = [point[0] + 0.1, point[1], point[2]]
        new_traj_data.append(new_point)

    return new_traj_data

class TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('trajectory_publisher')

        self.safe_distance = 1.5
        self.is_mpc_trajectory = True
        # 发布轨迹的话题
        self.trajectory_pub_ = self.create_publisher(PoseArray, 'trajectory', 10)
        self.mpc_trajectory_pub_ = self.create_publisher(PoseArray, 'mpc_trajectory', 10)
        self.obstacle_flag_pub_ = self.create_publisher(String, 'obstacle_reduce_speed', 1)

        # 订阅障碍物检测结果
        self.obstacle_image_sub_ = self.create_subscription(
            String,
            'image_detection_results',
            # self.obstacle_callback,
            self.parser_image_detection_results,
            10
        )
        
        # 订阅激光雷达检测结果
        self.obstacle_lidar_sub_ = self.create_subscription(
            String,
            'detection_results',
            self.parser_lidar_detection_results,
            # self.obstacle_callback,
            10
        )
        
        # 订阅基于规则激光雷达检测结果
        self.obstacle_lidar_rule_sub_ = self.create_subscription(
            String,
            'lidar_detections',
            # self.parser_rule_lidar_detection_results,
            self.obstacle_callback,
            10
        )

        if  not self.is_mpc_trajectory:
            # 读取 CSV 初始轨迹
            csv_file_path = '/home/nvidia/vcii/follow_trajectory/collect_trajectory/processed_campus_0411.csv'
            self.main_traj_data = read_csv(csv_file_path)

            # csv_file_path = '/home/nvidia/vcii/follow_trajectory/collect_trajectory/processed_lane_change_right_0404.csv'
            csv_file_path = '/home/nvidia/vcii/follow_trajectory/collect_trajectory/processed_campus_0411.csv'
            self.second_traj_data = read_csv(csv_file_path)
            self.follower = AdjustTrajectory(self.main_traj_data,
                                            self.second_traj_data)

            # 首次发布初始轨迹
            self.publish_trajectory(self.main_traj_data)
            # self.get_logger().info('Initial trajectory published.')
            
        else:
            #=============================下面是mpc================================
            # mpc_trajectory_csv = '/home/nvidia/vcii/follow_trajectory/collect_trajectory/processed_shiyanzhongxin_0327_with_yaw_ck.csv'
            mpc_trajectory_csv = '/home/nvidia/vcii/follow_trajectory/collect_trajectory/processed_campus_0411_with_yaw_ck.csv'
            self.main_traj_data = read_mpc_csv(mpc_trajectory_csv)  # 读取到的是一个list [[x,x,x,x,],[y,y,y,y,y],[yaw,yaw,yaw,...],[ck,ck,ck..]]
            self.publish_mpc_trajectory(self.main_traj_data)
        
            # mpc_trajectory_csv = '/home/nvidia/vcii/follow_trajectory/collect_trajectory/processed_shiyanzhongxin_0327_with_yaw_ck.csv'
            mpc_trajectory_csv = '/home/nvidia/vcii/follow_trajectory/collect_trajectory/processed_campus_0411_with_yaw_ck.csv'
            self.second_traj_data = read_mpc_csv(mpc_trajectory_csv)  # 读取到的是一个list [[x,x,x,x,],[y,y,y,y,y],[yaw,yaw,yaw,...],[ck,ck,ck..]]
        
            self.follower = AdjustTrajectory(self.main_traj_data,
                                             self.second_traj_data,
                                             is_mpc_trajectory = self.is_mpc_trajectory) 
    
        self.can_use = Can_use()
        # === 新增：定时器 ===
        # 周期性发布当前轨迹，无论是否收到障碍物信息
        self.timer_period = 0.1  # (HZ = 10)
        self.timer_ = self.create_timer(self.timer_period, self.timer_callback)

        #========== from rth lanechage class =========
        self.lane_change = LaneChangeDecider(cx   = self.main_traj_data[0],
                                             cy   = self.main_traj_data[1],
                                             cyaw = self.main_traj_data[2],
                                             ck   = self.main_traj_data[3],
                                            )
        self.state = State()
        
    def timer_callback(self):
        """
        定时器回调：每隔 self.timer_period 秒，将 latest trajectory 发布一次
        """
        current_traj_data = self.follower.current_trajectory
        if self.is_mpc_trajectory:
            self.publish_mpc_trajectory(current_traj_data)
        else:
            self.publish_trajectory(current_traj_data)
            
        
    def publish_trajectory(self, traj_data):
        """
        将轨迹数据（list of [x_utm, y_utm, heading(度数)]）发布为 PoseArray。
        注意：heading 是 0~360 度，需要转弧度再生成四元数。
        """
        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = self.get_clock().now().to_msg()
        pose_array_msg.header.frame_id = 'map'
        
        # 判断是否为None,如果为None,表示没有轨迹了，发一个空的pose_array_msg
        if traj_data is not None:
            for x_utm, y_utm, heading_deg in traj_data:
                pose = Pose()
                pose.position.x = float(x_utm)
                pose.position.y = float(y_utm)
                pose.position.z = 0.0

                # 将度数转换为弧度
                heading_rad = math.radians(heading_deg)
                qz = math.sin(heading_rad / 2.0)
                qw = math.cos(heading_rad / 2.0)
                pose.orientation.z = qz
                pose.orientation.w = qw

                pose_array_msg.poses.append(pose)
            self.get_logger().info('Published an  trajectory.')
                
        else:
            self.get_logger().info('Published an empty trajectory (no more path).')

        self.trajectory_pub_.publish(pose_array_msg)

    def publish_mpc_trajectory(self, traj_data):
        """
        traj_data: list of [cx, cy, cyaw(弧度), ck]
        """
        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = self.get_clock().now().to_msg()
        pose_array_msg.header.frame_id = 'map'

        if traj_data is not None:
            if len(traj_data) == 5:
                for cx, cy, yaw_rad, ck,sp in zip(traj_data[0],traj_data[1],traj_data[2],traj_data[3],traj_data[4]):
                    pose = Pose()
                    # 位置
                    pose.position.x = float(cx)
                    pose.position.y = float(cy)
                    pose.position.z = 0.0

                    # 航向: yaw 转四元数
                    # 只使用 z,w 表示绕 Z 轴旋转
                    qz = math.sin(yaw_rad / 2.0)
                    qw = math.cos(yaw_rad / 2.0)

                    # 将曲率 ck 暂时存在 orientation.x
                    # (或者 orientation.y，二选一)
                    pose.orientation.x = float(ck)

                    # orientation.y 可以置 0
                    pose.orientation.y = sp

                    pose.orientation.z = qz
                    pose.orientation.w = qw

                    pose_array_msg.poses.append(pose)
                    
            else:
                for cx, cy, yaw_rad, ck in zip(traj_data[0],traj_data[1],traj_data[2],traj_data[3]):
                    pose = Pose()
                    # 位置
                    pose.position.x = float(cx)
                    pose.position.y = float(cy)
                    pose.position.z = 0.0

                    # 航向: yaw 转四元数
                    # 只使用 z,w 表示绕 Z 轴旋转
                    qz = math.sin(yaw_rad / 2.0)
                    qw = math.cos(yaw_rad / 2.0)

                    # 将曲率 ck 暂时存在 orientation.x
                    # (或者 orientation.y，二选一)
                    pose.orientation.x = float(ck)

                    # orientation.y 可以置 0
                    pose.orientation.y = 0.0

                    pose.orientation.z = qz
                    pose.orientation.w = qw

                    pose_array_msg.poses.append(pose)
            # self.get_logger().info('Published an mpc trajectory.')
        else:
            self.get_logger().info('Published an empty trajectory.')

        self.mpc_trajectory_pub_.publish(pose_array_msg)

    def obstacle_callback(self, msg):
        """
        当订阅到障碍物信息时，决定是否要调用避障逻辑。
        如果 msg.data 为空或表示无障碍物，则直接发布原始轨迹。
        否则，发布调整后的轨迹。
        """
        # obstacle_list = self.parser_image_detection_results(msg)
        obstacle_list_utm, obstacle_list = self.parser_rule_lidar_detection_results(msg) # obstacle_list_utm utm坐标系障碍物坐标，obstacle_list局部坐标系坐标
        # [[x,y],[x,y]]
        # obstacle_list = self.parser_lidar_detection_results(msg)
        # 判断是否有障碍物
        # obstacle_list = []
        if len(obstacle_list) == 0:
            # 如果订阅内容为空字符串，或特定关键词表示“无障碍物”
            # 则发布“原始轨迹”即可
            # 没有障碍物 -> 发原轨迹
            self.obstacle_flag_pub_.publish(String(data="No"))
            # self.follower.current_trajectory = self.main_traj_data
            # self.get_logger().info('No obstacle detected, published main trajectory.')
        else:
            # 这里说明有障碍物信息，则调用调整轨迹
            self.obstacle_flag_pub_.publish(String(data="Yes"))
            # self.adjust_utm_trajectory(obstacle_list_utm)
            self.change_trajectory_rth(obstacle_list)  # rth avoid abstacle function
            # self.get_logger().info('Obstacle detected, published adjusted trajectory.')

    def parser_image_detection_results(self, msg):
        '''
        输入：ros 的msg
        输出：障碍物utm坐标
        '''
        for i in range(20):
            self.can_use.read_ins_info()
        data = msg.data
        # self.get_logger().info("接收到检测结果: " + data)
        obstacle_detected = False
        try:
            # 按照'|'分割，获取目标总数和坐标信息
            header, coordinates_str = data.split('|', 1)
            header = header.strip()  # 例如："Total targets: 3"
            total_targets = int(header.split(':')[1].strip())
            # self.get_logger().info("检测到目标总数: %d" % total_targets)

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
            obstacles_memoryBank = []
            obstacles_list_xy = []
            for one_obstacle in coordinates_list:
                obstacle_x = one_obstacle[2]
                obstacle_y = one_obstacle[0]
                # print("==========================",obstacle_x,obstacle_y)          
                if obstacle_x is not None and obstacle_y is not None and abs(obstacle_y) < 4 and abs(obstacle_x) < 20:
                    obstacle_x_utm, obstacle_y_utm = obstacle_position(
                        self.can_use.ego_lat, self.can_use.ego_lon, self.can_use.ego_yaw_deg, obstacle_x, obstacle_y
                    )
                    obstacles_memoryBank.append([obstacle_x_utm, obstacle_y_utm])
                    obstacles_list_xy.append([obstacle_x,obstacle_y])

            obstacles_list = obstacles_memoryBank.copy()
            for one in obstacles_list_xy:
                if  abs(one[0]) < 10 and abs(one[1]) < 4:
                    obstacle_detected = True
                    break
                else:
                    obstacle_detected = False
            
           
        except Exception as e:
            self.get_logger().error("解析检测结果失败: " + str(e))
        if obstacle_detected:
            return obstacles_list
        else:
            return []

    def parser_lidar_detection_results(self, msg):
        """
        回调函数：从订阅到的 String 中解析出 bounding box 信息。
        示例解析文本格式类似：
            检测到 2 个box:
              [0]: x=1.1, y=2.2, z=3.3, w=4.4, l=5.5, h=6.6, theta=0.7, score=0.95, label=car
              [1]: x=2.2, y=3.3, z=4.4, w=5.5, l=6.6, h=7.7, theta=1.1, score=0.90, label=truck
        """
        # 获取当前车辆信息
        for i in range(20):
            self.can_use.read_ins_info()
            
        data_str = msg.data
        # self.get_logger().info(f"收到原始消息:\n{data_str}")

        # 先解析出有多少个 box（可选，如果你需要此信息）
        first_line_pattern = r'检测到\s+(\d+)\s+个box'
        first_line_match = re.search(first_line_pattern, data_str)
        if not first_line_match:
            # self.get_logger().warn("未能解析到有效的box数量，检查字符串格式。")
            return
        
        num_boxes = int(first_line_match.group(1))
        # self.get_logger().info(f"解析到 box 数量: {num_boxes}")

        # 正则匹配每行内容：形如
        #   [i]: x=..., y=..., z=..., w=..., l=..., h=..., theta=..., score=..., label=...
        #
        # 注意：这里的正则假设浮点数里不会有空格等特殊情况，label 为单词字符。
        box_line_pattern = (
            r'\[(\d+)\]:\s*'
            r'x=([-\d.]+),\s*'
            r'y=([-\d.]+),\s*'
            r'z=([-\d.]+),\s*'
            r'w=([-\d.]+),\s*'
            r'l=([-\d.]+),\s*'
            r'h=([-\d.]+),\s*'
            r'theta=([-\d.]+),\s*'
            r'score=([-\d.]+),\s*'
            r'label=(\S+)'
        )

        # 用 findall 从整段文本中找出所有匹配
        matches = re.findall(box_line_pattern, data_str)
        if not matches:
            self.get_logger().warn("未能匹配到任何 bounding box 信息。")
            return

        # 输出各目标的坐标
        obstacles_memoryBank = []
        obstacles_list_xy = []
        for match in matches:
            # match 返回的是 (index, x, y, z, w, l, h, theta, score, label)，都还是字符串
            # 我们可以根据需要，把数值型的转换为 float 或 int
            idx, x_str, y_str, z_str, w_str, l_str, h_str, theta_str, score_str, label_str = match
            
            # 减4是为了做一个偏移量，原车标定不准
            obstacle_x, obstacle_y = float(x_str) - 4, -float(y_str)
            if obstacle_x < 10 and abs(obstacle_y) < 2:
                obstacle_x_utm, obstacle_y_utm = obstacle_position(
                    self.can_use.ego_lat, self.can_use.ego_lon, self.can_use.ego_yaw_deg, obstacle_x, obstacle_y
                )
                obstacles_memoryBank.append([obstacle_x_utm, obstacle_y_utm])
                obstacles_list_xy.append([obstacle_x,obstacle_y])
        # self.get_logger().info(f"youxiao {str(obstacles_list_xy)}")
    
        return obstacles_memoryBank
    
    def parser_rule_lidar_detection_results(self, msg):
        """
        回调函数：从订阅到的 String 中解析出 bounding box 信息。
        示例解析文本格式类似：
            输入形式：
        """
        obstacle_detected = False
        
        # 获取当前车辆信息
        for i in range(20):
            self.can_use.read_ins_info()
            
        data_str = msg.data.strip()
        # self.get_logger().info(f"收到原始消息:\n{data_str}")


        if data_str.endswith(";"):
            data_str = data_str[:-1]
        data_str = data_str.split(":")[-1]
        # print("data_str:",data_str)
        # 用分号把box切分开
        # data_str.split(';') -> ["x,y,z,w,l,h,yaw", "x,y,z,w,l,h,yaw", ...]
        box_str_list = data_str.split(';')
        
        # 用来存放最终的 [[x,y,z],[x,y,z],...] 结果
        # 输出各目标的坐标
        obstacles_memoryBank = []
        obstacles_list_xy = []
        
        for box_str in box_str_list:
            # 进一步用逗号分隔 -> ["x", "y", "z", "w", "l", "h", "yaw"]
            fields = box_str.split(',')
            if len(fields) < 3:
                continue  # 避免异常或格式不完整

            try:
                # 将前三项 (x, y, z) 转为 float
                obstacle_x = float(fields[0])
                obstacle_y = -float(fields[1])
                
                # 超过10m的直接放弃
                if obstacle_x < 20 and abs(obstacle_y) < 2:
                    obstacle_detected = True
                    
                    obstacle_x_utm, obstacle_y_utm = obstacle_position(
                        self.can_use.ego_lat, self.can_use.ego_lon, self.can_use.ego_yaw_deg, obstacle_x, obstacle_y
                    )
                    obstacles_memoryBank.append([obstacle_x_utm, obstacle_y_utm])
                    obstacles_list_xy.append([obstacle_x,obstacle_y])
            except ValueError:
                # 如果解析失败，可以打印日志或忽略
                self.get_logger().warn(f"无法解析这行数据: {box_str}")
        
        # 打印或保存 xyz_list
        # self.get_logger().info(f"提取出的坐标数组: {obstacles_list_xy}")
        if obstacle_detected:
            return obstacles_memoryBank, obstacles_list_xy
        else:
            return [],[]
  
    def adjust_trajectory(self, obstcle):
        
        # 在这里也可以做一些判断
        # 判断是否调整轨迹
        self.follower.check_and_switch_trajectory(obstcle,safe_distance=self.safe_distance)
        self.traj_data = self.follower.current_trajectory
        
    def adjust_utm_trajectory(self, obstcle):    
        # 在这里也可以做一些判断
        # 判断是否调整轨迹
        self.follower.check_and_switch_utm_trajectory(obstcle,safe_distance=self.safe_distance)
        self.traj_data = self.follower.current_trajectory
        

    def change_trajectory_rth(self,obstacle_list):
        # self.lane_change.init_refline()
        self.state.x   = self.can_use.ego_x
        self.state.y   = self.can_use.ego_y
        self.state.yaw = self.can_use.ego_yaw_rad
        self.state.v   = 2.7
        # print("=====obstacle_list:",obstacle_list)
        self.lane_change.update_state(self.state, obstacle_list)
        cx, cy, cyaw, ck, sp = self.lane_change.publish_new_refline()
        # self.mpc_main_traj_data = [cx,cy,cyaw,ck,sp]
        self.follower.current_trajectory = [cx,cy,cyaw,ck,sp]
    
def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped by user.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
