'''two path select'''
import can
from math import radians, cos, sin, asin, sqrt, degrees, atan2
import matplotlib.pyplot as plt
import numpy as np
import threading
from pynput import keyboard
import time
import math
# import cv2
import csv
from pyproj import Proj
import matplotlib.pyplot as plt
# from perception.yolov8_detect import CameraObjectDetector
from can_use import Can_use, ISGSpeedFilter

import pyproj
import time 

import logging
import datetime

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file_name = f"{timestamp}.log"

logging.basicConfig(
    filename=log_file_name,         # 日志输出到当前目录下的 <时间戳>.log 文件
    level=logging.INFO,             # 日志级别：INFO 及以上
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 车辆参数
VEHICLE_WIDTH = 1.9   # m
VEHICLE_LENGTH = 4.5  # m
WHEEL_FACTOR = 7.2
manual_triggered = False
stop_record = False
mod_666 = 0
mod_AE = 0

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
    plt.scatter(x , y)
    plt.scatter([x[-1]], [y[-1]], color="red") # end
    plt.scatter(12.27, -3.27, color="brown")
    plt.scatter([x[0]], [y[0]], color="black") # start
    plt.title('reference_trajectory_utm')  
    plt.xlabel('longitudinal')  
    plt.ylabel('latitudinal')
    plt.savefig('ref_traj_utm.png')
    return traj_data


# 计算轨迹点与障碍物的距离
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_offset_direction(obstacles_distance, obstacle_idx):
    directions = 0
    for idx in obstacle_idx:
        if obstacles_distance[idx][1] > 0:
            directions += 1
        else:
            directions += -1
    if directions >= 0:
        avg_direction = 1
    else:
        avg_direction = -1
    return avg_direction

# 将距离较近的障碍物分为一组
def group_nearby_obstacles(obstacles, distance_threshold=5):
    groups = []
    obstacle_idxs = []
    for i, obstacle in enumerate(obstacles):
        # 检查障碍物是否已经被分配到某一组
        found_group = False
        for group, obstacle_idx in zip(groups, obstacle_idxs):
            # 如果障碍物与该组内的任意障碍物距离小于阈值，加入该组
            if any(calculate_distance(obstacle, existing_obstacle) < distance_threshold for existing_obstacle in group):
                group.append(obstacle)
                found_group = True
                obstacle_idx.append(i)
                break
        if not found_group:
            # 如果没有找到相近的组，则创建一个新组
            groups.append([obstacle])
            obstacle_idxs.append([i])
    return groups, obstacle_idxs

# 计算障碍物组的中心位置（质心）
def calculate_group_center(group):
    group_center = np.mean(group, axis=0)
    return group_center

# 初始化 WGS84 到 UTM 的投影转换器
wgs84 = pyproj.CRS("EPSG:4326")
utm = pyproj.CRS("EPSG:32650")  # 根据经度选择合适的 UTM 区域，这里用 50 区为例
projector_to_utm = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True)
projector_to_wgs84 = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True)

def latlon_to_utm(lon, lat):
    """将经纬度转换为 UTM 坐标"""
    x, y = projector_to_utm.transform(lon, lat)
    return x, y

def utm_to_latlon(x, y):
    """将 UTM 坐标转换为经纬度"""
    lon, lat = projector_to_wgs84.transform(x, y)
    return lon, lat

def offset_point_utm(x, y, heading, offset_distance, direction='left'):
    heading_rad = math.radians(heading)
    offset_angle = heading_rad + (math.pi / 2 if direction == 'left' else -math.pi / 2)
    dx = offset_distance * math.cos(offset_angle)
    dy = offset_distance * math.sin(offset_angle)
    return x + dx, y + dy


def obstacle_position(lat0, lon0, heading, x, y):
    """
    计算障碍物的经纬度
    :param lat0: 车辆的纬度（度）
    :param lon0: 车辆的经度（度）
    :param heading: 车辆的航向角（度）
    :param x: 障碍物相对于车辆的纵向距离（米），向前为正
    :param y: 障碍物相对于车辆的横向距离（米），向右为正
    :return: 障碍物的 UTM 坐标（x, y）
    """
    # 将航向角转换为弧度
    theta = math.radians(heading)

    # 计算障碍物相对于车辆的位置（东向和北向）
    delta_E = x * math.sin(theta) + y * math.cos(theta)
    delta_N = x * math.cos(theta) - y * math.sin(theta)

    # 将车辆当前位置转换为 UTM 坐标
    x0, y0 = latlon_to_utm(lon0, lat0)

    # 计算障碍物的 UTM 坐标
    obstacle_x = x0 + delta_E
    obstacle_y = y0 + delta_N

    # 返回障碍物的 UTM 坐标
    return obstacle_x, obstacle_y
# 

# 判断当前障碍物是否需要被忽略
def should_ignore_obstacle(current_obstacle, known_obstacles, distance_threshold=5.0):
    """
    判断当前障碍物是否需要忽略。当前障碍物与已知障碍物距离小于阈值则认为已经处理过。
    :param current_obstacle: 当前障碍物坐标，格式为 (x, y)
    :param known_obstacles: 已知障碍物坐标集合，格式为 [[x1, y1], [x2, y2], ...]
    :param distance_threshold: 判断是否接近的距离阈值，单位米
    :return: 如果距离阈值内存在已知障碍物，则返回True，表示忽略；否则返回False
    """
    if len(known_obstacles) == 0:
        return False
    
    for obstacle in known_obstacles:
        dist = calculate_distance(current_obstacle, obstacle)
        if dist < distance_threshold:
            return True  # 如果当前障碍物与任何已知障碍物距离小于阈值，则忽略
    return False  # 否则，不忽略

class VehicleTrajectoryFollower:
    def __init__(self, main_trajectory_csv, alternate_trajectory_csv, target_index=0):
        """
        初始化，读取主轨迹和备选轨迹点
        :param main_trajectory_csv: 包含主轨迹点的CSV文件路径
        :param alternate_trajectory_csv: 包含备选轨迹点的CSV文件路径
        """
        self.main_trajectory = read_csv(main_trajectory_csv)
        self.alternate_trajectory = read_csv(alternate_trajectory_csv)
        self.current_trajectory = self.main_trajectory.copy()
        self.is_using_alternate = False  # 标志当前是否在使用备选轨迹
        self.main_closest_index = 0
        self.alternate_closest_index = 0
        self.wheelbase = 3.5
        self.offset_target_index = 5
        self.target_index = 0
        self.should_stop = False  # 增加停车标志位
        self.obstacle_detected = False  # 标记是否检测到障碍物
        self.previous_turn_angle = 0
        self.far_previous_turn_angle = 0
        self.max_turn_rate = 6  # 限制每次转向角的最大变化速率（度）
        self.far_index = 25  # 远处的目标点，用于控制速度
        self.control_speed_the = 30 #用於判斷遠處目標點和當前head的差值是否超過該值，然後進行速度的處理
    
    def update_closest_indices(self, current_lat, current_lon):
        self.main_closest_index = self.find_closest_point_index(
            current_lat, current_lon, self.main_trajectory, is_main=True
        )
        self.alternate_closest_index = self.find_closest_point_index(
            current_lat, current_lon, self.alternate_trajectory, is_main=False
        )    
    
    def switch_to_alternate(self):
        """切换到备选轨迹"""
        print("切换到备选轨迹")
        self.current_trajectory = self.alternate_trajectory.copy()
        self.is_using_alternate = True

    def switch_to_main(self):
        """切换回主轨迹"""
        print("切换回主轨迹")
        self.current_trajectory = self.main_trajectory.copy()
        self.is_using_alternate = False

    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        dLon = math.radians(lon2 - lon1)
        lat1 = math.radians(lat1)
        lat2 = math.radians(lat2)
        x = math.sin(dLon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dLon))
        initial_bearing = math.atan2(x, y)
        initial_bearing = math.degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360
        return compass_bearing

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        R = 6371000  # 地球半径，单位为米
        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)
        a = math.sin(dLat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance
    
    
    def find_closest_point_index_bank(self, current_lat, current_lon, trajectory=None):
        """
        找到距离当前车辆位置最近的轨迹点索引
        :param current_lat: 车辆当前纬度
        :param current_lon: 车辆当前经度
        :param trajectory: 要查找的轨迹，默认使用当前轨迹
        :return: 距离最近的轨迹点索引
        """
        if trajectory is None:
            trajectory = self.current_trajectory
        closest_index = 0
        min_distance = float('inf')
        if self.closest_index == 0:
            max_bound = len(trajectory)-1
        else:
            max_bound = 200
        for i, (lon, lat, _) in enumerate(trajectory[self.closest_index:self.closest_index+max_bound]):  # 经度在前，纬度在后
            distance = self.calculate_distance(current_lat, current_lon, lat, lon)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        return closest_index+self.closest_index

    
    def find_closest_point_index(self, current_lat, current_lon, trajectory=None,is_main=True):
        """
        找到距离当前车辆位置最近的轨迹点索引
        :param current_lat: 车辆当前纬度
        :param current_lon: 车辆当前经度
        :param trajectory: 要查找的轨迹，默认使用当前轨迹
        :return: 距离最近的轨迹点索引
        """
        if trajectory is None:
            trajectory = self.current_trajectory
        closest_index_temp = 0
        min_distance = float('inf')
        
        if is_main:
            closest_index = self.main_closest_index
        else:
            closest_index = self.alternate_closest_index
        
        if closest_index == 0:
            max_bound = len(trajectory)-1
        else:
            max_bound = 200
        print(max(closest_index,0),min(closest_index+max_bound, len(trajectory)-1))
        for i, (lon, lat, _) in enumerate(trajectory[closest_index:max(closest_index+max_bound,len(trajectory))]):  # 经度在前，纬度在后
            distance = self.calculate_distance(current_lat, current_lon, lat, lon)
            if distance < min_distance:
                min_distance = distance
                closest_index_temp = i
                
        return closest_index_temp+closest_index
    
    def find_closest_point_index_avoid(self, current_lat, current_lon, trajectory=None):
        """
        找到距离当前车辆位置最近的轨迹点索引
        :param current_lat: 车辆当前纬度
        :param current_lon: 车辆当前经度
        :param trajectory: 要查找的轨迹，默认使用当前轨迹
        :return: 距离最近的轨迹点索引
        """
        if trajectory is None:
            trajectory = self.current_trajectory

        min_distance = float('inf')
        max_bound = len(trajectory)-1
        for i, (lon, lat, _) in enumerate(trajectory):  # 经度在前，纬度在后
            distance = self.calculate_distance(current_lat, current_lon, lat, lon)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        return closest_index
    
    def adjust_position_to_front_axle(self, rear_lat, rear_lon, heading):
        """
        根据后轴中心的经纬度和heading计算前轴的经纬度
        :param rear_lat: 后轴的纬度
        :param rear_lon: 后轴的经度
        :param heading: 车辆的航向角，相对于正北方向
        :return: 前轴的经纬度 (lat, lon)
        """
        # 先将heading转换为弧度
        heading_rad = math.radians(heading)

        # 计算纬度上的变化，假设1度纬度大约为111,320米
        delta_lat = (self.wheelbase / 6371000) * math.cos(heading_rad)

        # 计算经度上的变化，假设经度的变化随着纬度而变化，纬度越高，1度经度的实际距离越小
        delta_lon = (self.wheelbase / 6371000) * math.sin(heading_rad) / math.cos(math.radians(rear_lat))

        # 计算前轴的经纬度
        front_lat = rear_lat + math.degrees(delta_lat)
        front_lon = rear_lon + math.degrees(delta_lon)

        return front_lat, front_lon
    
    # 用于平滑转角
    def smooth_turn_angle(self, turn_angle):
        # 限制转向角的最大变化速率
        angle_difference = turn_angle - self.previous_turn_angle
        if angle_difference > self.max_turn_rate:
            update_turn_angle = self.previous_turn_angle + 4
        elif angle_difference < -self.max_turn_rate:
            update_turn_angle = self.previous_turn_angle - 4
        else:
            update_turn_angle = turn_angle
        
    

        # 更新上一次的转向角
        self.previous_turn_angle = update_turn_angle
        return turn_angle

    def calculate_turn_angle(self, current_position, current_heading,offset_target_index = None):
        if offset_target_index is not None:
            target_index_obstacle = offset_target_index
        else:
            target_index_obstacle = self.offset_target_index
        print("==============", target_index_obstacle)
        current_lat, current_lon, _ = current_position
        # 根据后轴的位置和heading调整得到前轴的位置
        front_lat, front_lon = self.adjust_position_to_front_axle(current_lat, current_lon, current_heading)
        
        # 找到距离最近的点的索引
        if self.is_using_alternate:
            # closest_index = self.find_closest_point_index(front_lat, front_lon)
            closest_index = self.alternate_closest_index
        else:
            closest_index = self.main_closest_index
        self.closest_index = closest_index
        print(self.closest_index )
        
        target_index = min(self.closest_index + target_index_obstacle, len(self.current_trajectory) - 1)  # 防止超出范围
        self.target_index = target_index
        next_lon, next_lat, _ = self.current_trajectory[target_index]  # 注意经纬度顺序
        
        # 计算目标点相对当前位置的方位角
        desired_heading = self.calculate_bearing(current_lat, current_lon, next_lat, next_lon)
        # 计算转向角
        turn_angle = (desired_heading - current_heading + 360) % 360
        if turn_angle > 180:
            turn_angle -= 360        
        # 映射到方向盘转角
        if turn_angle * WHEEL_FACTOR > 460:
            turn_angle = 460
        elif turn_angle * WHEEL_FACTOR < -460:
            turn_angle = -460
        else:
            turn_angle = turn_angle * WHEEL_FACTOR    
        
        turn_angle = self.smooth_turn_angle(turn_angle)
        
        return turn_angle

 
    # 计算期望速度和加速度
    def calculate_speedAndacc(self, turn_angle, current_position, current_speed, is_obstacle = False, points_num_threshold=20):
        if current_speed < 1:
            speed = 20
            acc = 0
            return speed, acc
        if abs(turn_angle) >= 50:
            if current_speed >= 15:
                speed = 10
                acc = -2
            else:
                speed = 10
                acc = 0
            return speed, acc 
        
        current_lat, current_lon, current_heading = current_position
        next_lon, next_lat, _ = self.current_trajectory[min(self.closest_index + self.far_index, len(self.current_trajectory) - 1)]  # 注意经纬度顺序
        
        # 计算目标点相对当前位置的方位角
        far_desired_heading = self.calculate_bearing(current_lat, current_lon, next_lat, next_lon)
        # 计算转向角
        far_turn_angle = (far_desired_heading - current_heading + 360) % 360
        if far_turn_angle > 180:
            far_turn_angle -= 360        
        # 映射到方向盘转角
        if far_turn_angle * WHEEL_FACTOR > 460:
            far_turn_angle = 460
        elif far_turn_angle * WHEEL_FACTOR < -460:
            far_turn_angle = -460
        else:
            far_turn_angle = far_turn_angle * WHEEL_FACTOR    

        if abs(far_turn_angle) >= 40:
            print("=_="*30)
            if current_speed >= 15:
                speed = 10
                acc = -1
            else:
                speed = 10
                acc = 0
        else:
            speed = 20
            acc = 0
        
        if is_obstacle:
            print("find obstacle reduce speed")
            if current_speed >= 15:
                speed = 10
                acc = -3
            else:
                speed = 10
                acc = 0
        return speed, acc    

    def check_trajectory_safe(self, trajectory, obstacles, safe_distance):
        """
        检查给定轨迹是否安全
        :param trajectory: 要检查的轨迹
        :param obstacles: 障碍物列表
        :param safe_distance: 安全距离
        :return: 如果轨迹安全返回 True，否则返回 False
        """
        if len(obstacles) == 0:
            # print("len is 0")
            return True
        for obstacle in obstacles:
            obstacle_x, obstacle_y = obstacle[0], obstacle[1]
            obstacle_lon, obstacle_lat = utm_to_latlon(obstacle_x, obstacle_y)
            closest_point_idx = self.find_closest_point_index_avoid(obstacle_lat, obstacle_lon, trajectory=trajectory)
            print(closest_point_idx)
            traj_point_lon, traj_point_lat, _ = trajectory[closest_point_idx]
            dist_to_closest_point = self.calculate_distance(obstacle_lat, obstacle_lon, traj_point_lat, traj_point_lon)
            print("======================distance:===================",dist_to_closest_point)
            if dist_to_closest_point < safe_distance:
                return False
        return True

    def check_and_switch_trajectory(self, obstacles, safe_distance=4):
        """
        检查当前轨迹是否安全，并在必要时进行轨迹切换或停车
        :param obstacles: 障碍物的坐标列表，格式为 [[x, y], ...]
        :param safe_distance: 安全距离
        """
        main_safe = self.check_trajectory_safe(self.main_trajectory, obstacles, safe_distance)
        alternate_safe = self.check_trajectory_safe(self.alternate_trajectory, obstacles, safe_distance)
        # print(main_safe,alternate_safe)
        if not main_safe and not alternate_safe:
            # 两条轨迹都不安全，设置停车标志
            print("主轨迹和备选轨迹都被阻塞，车辆需要停止")
            self.should_stop = True
        elif not self.is_using_alternate and not main_safe and alternate_safe:
            # 主轨迹不安全，备选轨迹安全，切换到备选轨迹
            self.switch_to_alternate()
            self.should_stop = False
        elif self.is_using_alternate and not alternate_safe and main_safe:
            # 备选轨迹不安全，主轨迹安全，切换回主轨迹
            self.switch_to_main()
            self.should_stop = False
        elif self.is_using_alternate and main_safe and alternate_safe:
            self.switch_to_main()
            self.should_stop = False
        elif not self.is_using_alternate and main_safe:
            # 继续使用主轨迹
            self.should_stop = False
        elif self.is_using_alternate and alternate_safe:
            # 继续使用备选轨迹
            self.should_stop = False
        else:
            # 其他情况，保持当前轨迹
            pass

class Can_use:
    def __init__(self, zone):
        self.bus_ins = can.interface.Bus(channel='can0', bustype='socketcan')
        self.bus_vcu = can.interface.Bus(channel='can1', bustype='socketcan')
        self.ego_lon = 31.8925019
        self.ego_lat = 118.8171577
        self.ego_yaw = 270
        self.ego_v = 3
        self.ego_a = 0
        self.eps_mode = 2
        self.auto_driver_allowed = False

    def read_ins_info(self):
        """获取惯导的主车信息"""
        message_ins = self.bus_ins.recv()
        message_vcu = self.bus_vcu.recv()
        if message_ins is not None and message_ins.arbitration_id == 0x504:
            # 直接获取数据字节
            can_data = message_ins.data
            # 解析前4个字节为纬度
            INS_Latitude = (can_data[0] << 24) | (can_data[1] << 16) | (can_data[2] << 8) | can_data[3]
            # 解析后4个字节为经度
            INS_Longitude = (can_data[4] << 24) | (can_data[5] << 16) | (can_data[6] << 8) | can_data[7]
            INS_Latitude = INS_Latitude * 0.0000001 - 180
            INS_Longitude = INS_Longitude * 0.0000001 - 180

            ego_x = INS_Longitude
            ego_y = INS_Latitude
            self.ego_lon = ego_x
            self.ego_lat = ego_y
            logging.info(f"ego_x:{ego_x},ego_y:{ego_y}")

        if message_ins is not None and message_ins.arbitration_id == 0x505:
            speed_data = message_ins.data
            # 北向速度
            INS_NorthSpd =  (speed_data[0] << 8) | speed_data[1]
            INS_NorthSpd =   INS_NorthSpd * 0.0030517 - 100    # m/s
            INS_NorthSpd *= 3.6
            # 东向速度
            INS_EastSpd =  (speed_data[2] << 8) | speed_data[3]
            INS_EastSpd =   INS_EastSpd * 0.0030517 - 100    # m/s
            INS_EastSpd *= 3.6
            # 地向速度
            INS_ToGroundSpd =  (speed_data[4] << 8) | speed_data[5]
            INS_ToGroundSpd =   INS_ToGroundSpd * 0.0030517 - 100    # m/s
            INS_ToGroundSpd *= 3.6
                    
            speed =  sqrt(INS_EastSpd**2 + INS_NorthSpd**2 + INS_ToGroundSpd**2)
                    
            self.ego_v = speed

        if message_ins is not None and message_ins.arbitration_id == 0x502:
            # self.ego_yaw = angle
            Angle_data = message_ins.data
            HeadingAngle =  (Angle_data[4] << 8) | Angle_data[5]
            HeadingAngle = HeadingAngle * 0.010986 - 360
            self.ego_yaw = HeadingAngle 
        if message_ins is not None and message_ins.arbitration_id == 0x500:
            acc_data = message_ins.data
            # 北向速度
            ACC_X =  (acc_data[0] << 8) | acc_data[1]
            ACC_X =   (ACC_X * 0.0001220703125 - 4) * 9.8   # g
            self.ego_a = ACC_X
        
        if message_vcu is not None and message_vcu.arbitration_id == 0x15C:
            allow_value = message_vcu.data[2] & 0x01
            self.auto_driver_allowed = (allow_value == 1)

        if message_vcu is not None and message_vcu.arbitration_id == 0x124:
            eps_mode = (message_vcu.data[6] >> 4) & 0x03
            self.eps_mode = eps_mode

    def publish_planner_action(self, action, id, action_type, mod, enable):
        """将规划动作发布到CAN"""

        if action_type == "angle":    
            # 数据缩放和转换
            data1 = int((action - (-738)) / 0.1)  # 确保data1根据传入angle正确计算
            data1_high = (data1 >> 8) & 0xFF    # data1的高8位
            data1_low = data1 & 0xFF            # data1的低8位

            data2 = int(mod) & 0x03             # data2缩放到2位范围，0-3
            data3 = int(250 / 10) & 0xFF     # data3缩放到8位范围，0-255, angle_spd=100
            data4 = int(enable) & 0x01          # data4缩放到1位范围，0或1
                
            # 构建发送数据，确保8字节长度
            data = [data1_high, data1_low, data2, data3, data4, 0, 0, 0]

            msg = can.Message(arbitration_id=id, data=data, is_extended_id=False)
            self.bus_vcu.send(msg)
        
        if action_type == "acc":
            auto_drive_cmd_bits = mod & 0x07  # 取最低3位
            # Auto speed cmd（位3-7）
            # 首先对速度进行缩放和偏移
            # 期望速度 单位km/h
            # desired_speed = action[0] 
            desired_speed = 3
            speed_scaled = int(desired_speed) & 0x1F  # 取5位（位3-7）
            # 组合BYTE0
            byte0 = (speed_scaled << 3) | auto_drive_cmd_bits

            # BYTE1-BYTE2（需求方向盘转角）
            # 需要根据具体缩放因子和偏移量进行计算，假设缩放因子为0.1，偏移量为0
            # action[1] = 396
            logging.info(f"final turn angle:{action[1]}")
            angle_scaled = int((action[1] - (-500)) / 0.1) & 0xFFFF  # 16位
            byte1 = (angle_scaled >> 8) & 0xFF  # 高8位
            byte2 = angle_scaled & 0xFF         # 低8位

            # BYTE3（需求制动减速度）
            # 进行缩放和偏移
            acc  =  action[2]
            # acc = 0
            acc_scaled = int((acc - (-4)) / 1) & 0xFF  # 假设缩放因子1，偏移量-4

            # 构建发送数据，剩余字节填充0
            data_666 = [byte0, byte1, byte2, acc_scaled, 0, 0, 0, 0]
            
            msg = can.Message(arbitration_id=id, data=data_666, is_extended_id=False)
            # 发送CAN消息
            self.bus_vcu.send(msg)
            # time.sleep(0.01)


def on_press(key):
    global manual_triggered
    global stop_record
    try:
        if key.char == 's':
            manual_triggered = True
            print("收到键盘输入's'，手动请求自动驾驶模式")
        if key.char == 'q':
            manual_triggered = False
        if key.char == "x":
            stop_record = True
    except AttributeError:
        if key == keyboard.Key.esc:
            print("收到Esc键，退出程序")
            return False  # 停止监听


def keyboard_listener():  
    # 创建并启动键盘监听器线程  
    with keyboard.Listener(on_press=on_press) as listener:  
        listener.join() 


def start_main(shared_data, can_use, follower, filter):
    global mod_666
    global mod_AE
    global manual_triggered
    obstacles_memoryBank = []  # 存储障碍物UTM坐标
    obstacles_list_xy = []
    obstacle_detected = False
    can_use.read_ins_info()
    while True:
        obstacle_detected = False
        can_use.read_ins_info()
        if can_use.eps_mode != 3 and manual_triggered:
            mod_AE = 1
            mod_666 = 1
        if can_use.eps_mode == 3:
            mod_AE = 3
            mod_666 = 0
            manual_triggered = False
        if mod_AE == 1 and mod_666 == 1:
            if can_use.ego_lon is not None and can_use.ego_lat is not None:
                follower.update_closest_indices(can_use.ego_lat, can_use.ego_lon)
                with shared_data["lock"]:
                    if shared_data["frame"]['perception_result'] is not None:
                        last_perception_frame = shared_data["frame"]['perception_result']
                        print("last_perception_framelast_perception_framelast_perception_framelast_perception_framelast_perception_frame",last_perception_frame)
                        obstacles_memoryBank = []
                        obstacles_list_xy = []
                        for one_obstacle in last_perception_frame:
                            obstacle_x = one_obstacle[2]
                            obstacle_y = one_obstacle[0]
                            # print("==========================",obstacle_x,obstacle_y)          
                            if obstacle_x is not None and obstacle_y is not None and abs(obstacle_y) < 4 and abs(obstacle_x) < 20:
                                obstacle_x_utm, obstacle_y_utm = obstacle_position(
                                    can_use.ego_lat, can_use.ego_lon, can_use.ego_yaw, obstacle_x, obstacle_y
                                )
                                obstacles_memoryBank.append([obstacle_x_utm, obstacle_y_utm])
                                obstacles_list_xy.append([obstacle_x,obstacle_y])
                    
                # 更新障碍物列表
                obstacles_list = obstacles_memoryBank.copy()
                for one in obstacles_list_xy:
                    if  abs(one[0]) < 50 and abs(one[1]) < 4:
                        obstacle_detected = True
                        break
                    else:
                        obstacle_detected = False
                # 检查并切换轨迹或停车
                # follower.check_and_switch_trajectory(obstacles_list, safe_distance=3)

                if follower.should_stop:
                    # 如果需要停车，发送停车指令
                    print("车辆停止")
                    new_frame = [0, 0, -1]  # 设置速度为0，加速度为-1表示制动
                else:
                    # 正常计算控制指令
                    if obstacle_detected:
                        turn_angle = follower.calculate_turn_angle(
                            (can_use.ego_lat, can_use.ego_lon, can_use.ego_yaw), can_use.ego_yaw, offset_target_index=3
                        )
                        # turn_angle /= 5
                    else:
                        turn_angle = follower.calculate_turn_angle(
                            (can_use.ego_lat, can_use.ego_lon, can_use.ego_yaw), can_use.ego_yaw
                        )
                    desired_speed, desired_acc = follower.calculate_speedAndacc(
                        turn_angle, (can_use.ego_lat, can_use.ego_lon, can_use.ego_yaw), can_use.ego_v, is_obstacle = obstacle_detected
                    )
                    filtered_angle = filter.update_speed(turn_angle)
                    logging.info(f'trun angle: {turn_angle}, filter angle: {filtered_angle}')
                    new_frame = [desired_speed, filtered_angle, desired_acc]
                # print(time.time()-t0)
            else:
                print("主车定位丢失...")
                new_frame = [0, 0, 0]
        else:
            print("请按s进入自动驾驶模式...")
            new_frame = [0, 0, 0]
            continue
        with shared_data['lock']:
            shared_data['frame']['control_result'] = new_frame

def start_detect(shared_data, model, camera_index):
    try:
         # 初始化相机
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise Exception("=========================无法打开摄像头=========================")
        print("=========================成功打开相机==============================")
        while True:
            ret, camera_frame = cap.read()
            if not ret: 
                print("===========================无法获取帧，结束！=============================")
                break
            detect_result = model.detect_objects(camera_frame)
            # print("==============:::::::::::::::::::::::::::", detect_result)
            with shared_data['lock']:
                shared_data['frame']['perception_result'] = detect_result
    except KeyboardInterrupt:
        print("检测停止")
    finally:
        cap.release()

def send_frame(shared_data, can_use):
    last_control_frame = None
    global mod_AE
    while True:
        # 每隔0.01秒发送一帧（100帧每秒）
        time.sleep(0.005)
        # 使用锁来读取共享数据
        with shared_data["lock"]:
            if shared_data["frame"]['control_result'] is not None:
                last_control_frame = shared_data["frame"]['control_result']
        
        if last_control_frame is not None:
            # print(last_control_frame)
            can_use.publish_planner_action(action=last_control_frame, id=0x666, action_type="acc", mod=1, enable=1)
                              
def main():
    # 使用示例
    camera_index = 0
    main_trajectory_csv = '/home/nvidia/vcii/follow_trajectory/collect_trajectory/processed_shiyanzhongxin_0327.csv'
    alternate_trajectory_csv = '/home/nvidia/vcii/follow_trajectory/collect_trajectory/processed_haima-1119-right.csv'
    follower = VehicleTrajectoryFollower(main_trajectory_csv, alternate_trajectory_csv, target_index=5)
    # 创建过滤器实例
    filter = ISGSpeedFilter()
    # 初始化canbus
    can_use = Can_use(zone=49)
 
    # 初始化相机检测
    camera_detector = CameraObjectDetector(model_path='perception/yolov8s.pt')
    # 用于在线程之间共享数据
    temp_result = {
        "control_result": None,
        "perception_result": None
    }
    shared_data = {
        "frame": temp_result,  # 存储最新的帧
        "lock": threading.Lock()  # 用于保证线程安全的锁
    }
    
    # 创建计算线程和发送线程
    compute_thread = threading.Thread(target=start_main, args=(shared_data, can_use, follower, filter))
    send_thread = threading.Thread(target=send_frame, args=(shared_data, can_use))
    detect_thread = threading.Thread(target=start_detect, args=(shared_data, camera_detector, camera_index))
    
    # 键盘监听线程
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)

    # 启动线程
    keyboard_thread.start()
    compute_thread.start()
    send_thread.start()
    detect_thread.start()
        
    # 主线程等待计算和发送线程完成（通常不会退出）
    compute_thread.join()
    send_thread.join()
    keyboard_thread.join()
    detect_thread.join()
        
if __name__ == '__main__':
    main()
