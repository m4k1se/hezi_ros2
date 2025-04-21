'''two path select'''
import can
from math import radians, cos, sin, asin, sqrt, degrees, atan2
import matplotlib.pyplot as plt
import numpy as np
import threading
import time
import math
# import cv2
import csv
from pyproj import Proj
import matplotlib.pyplot as plt
# from perception.yolov8_detect import CameraObjectDetector
from can_use import Can_use

import pyproj
import time 

# import logging
# import datetime

# timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
# log_file_name = f"{timestamp}.log"

# logging.basicConfig(
#     filename=log_file_name,         # 日志输出到当前目录下的 <时间戳>.log 文件
#     level=logging.INFO,             # 日志级别：INFO 及以上
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

# 车辆参数
VEHICLE_WIDTH = 1.9   # m
VEHICLE_LENGTH = 4.5  # m
WHEEL_FACTOR = 7.2
manual_triggered = False
stop_record = False
mod_666 = 0
mod_AE = 0

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

class AdjustTrajectory:
    def __init__(self, main_trajectory_csv, alternate_trajectory_csv, target_index=0, is_mpc_trajectory=False):
        """
        初始化，读取主轨迹和备选轨迹点
        :param main_trajectory_csv: 包含主轨迹点的CSV文件路径
        :param alternate_trajectory_csv: 包含备选轨迹点的CSV文件路径
        """
        self.main_trajectory = main_trajectory_csv
        self.alternate_trajectory = alternate_trajectory_csv
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

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        R = 6371000  # 地球半径，单位为米
        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)
        a = math.sin(dLat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance
    
    # 计算utm坐标点之间的距离
    def calculate_utm_distance(self,x1, y1, x2, y2):
        """
        计算两个 UTM 坐标点之间的欧几里得距离，单位：米
        """
        dx = x2 - x1
        dy = y2 - y1
        return math.sqrt(dx * dx + dy * dy) 
    
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

    def find_utm_closest_point_index_avoid(self, current_lat, current_lon, trajectory=None):
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
        for i, (ref_utm_x, ref_utm_y) in enumerate(zip(trajectory[0],trajectory[1])):  # 经度在前，纬度在后
            distance = self.calculate_utm_distance(current_lat, current_lon, ref_utm_x, ref_utm_y)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        return closest_index

    
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
    
    def check_utm_trajectory_safe(self, trajectory, obstacles, safe_distance):
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
            closest_point_idx = self.find_utm_closest_point_index_avoid(obstacle_x, obstacle_y, trajectory=trajectory)
            # print(closest_point_idx)
            traj_point_utm_x, traj_point_utm_y = trajectory[0][closest_point_idx],trajectory[1][closest_point_idx]
            dist_to_closest_point = self.calculate_utm_distance(obstacle_x, obstacle_y, traj_point_utm_x, traj_point_utm_y)
            print("======================distance:===================",dist_to_closest_point)
            if dist_to_closest_point < safe_distance:
                return False
        return True
    
    def check_and_switch_trajectory(self, obstacles, safe_distance=2):
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
            self.current_trajectory = None
        elif not self.is_using_alternate and not main_safe and alternate_safe:
            # 主轨迹不安全，备选轨迹安全，切换到备选轨迹
            print("轨迹不安全，备选轨迹安全，切换到备选轨迹")
            self.switch_to_alternate()
            self.should_stop = False
        elif self.is_using_alternate and not alternate_safe and main_safe:
            # 备选轨迹不安全，主轨迹安全，切换回主轨迹
            print("备选轨迹不安全，主轨迹安全，切换回主轨迹")
            self.switch_to_main()
            self.should_stop = False
        elif self.is_using_alternate and main_safe and alternate_safe:
            print("都安全，切换回主轨迹")
            self.switch_to_main()
            self.should_stop = False
        elif not self.is_using_alternate and main_safe:
            # 继续使用主轨迹
            print("继续使用主轨迹")
            self.should_stop = False
        elif self.is_using_alternate and alternate_safe:
            # 继续使用备选轨迹
            print("继续使用备选轨迹")
            self.should_stop = False
        else:
            # 其他情况，保持当前轨迹
            print("保持轨迹不变")
            pass

    def check_and_switch_utm_trajectory(self, obstacles, safe_distance=2):
        """
        检查当前轨迹是否安全，并在必要时进行轨迹切换或停车
        :param obstacles: 障碍物的坐标列表，格式为 [[x, y], ...]
        :param safe_distance: 安全距离
        """
        
        main_safe = self.check_utm_trajectory_safe(self.main_trajectory, obstacles, safe_distance)
        alternate_safe = self.check_utm_trajectory_safe(self.alternate_trajectory, obstacles, safe_distance)
        # print(main_safe,alternate_safe)
        if not main_safe and not alternate_safe:
            # 两条轨迹都不安全，设置停车标志
            print("主轨迹和备选轨迹都被阻塞，车辆需要停止")
            self.should_stop = True
            self.current_trajectory = None
        elif not self.is_using_alternate and not main_safe and alternate_safe:
            # 主轨迹不安全，备选轨迹安全，切换到备选轨迹
            print("mpc轨迹不安全，备选mpc轨迹安全，切换到备选mpc轨迹")
            self.switch_to_alternate()
            self.should_stop = False
        elif self.is_using_alternate and not alternate_safe and main_safe:
            # 备选轨迹不安全，主轨迹安全，切换回主轨迹
            print("备选mpc轨迹不安全，主轨迹安全，切换回主mpc轨迹")
            self.switch_to_main()
            self.should_stop = False
        elif self.is_using_alternate and main_safe and alternate_safe:
            print("都安全，切换回主mpc轨迹")
            self.switch_to_main()
            self.should_stop = False
        elif not self.is_using_alternate and main_safe:
            # 继续使用主轨迹
            print("继续使用主mpc轨迹")
            self.should_stop = False
        elif self.is_using_alternate and alternate_safe:
            # 继续使用备选轨迹
            print("继续使用备选mpc轨迹")
            self.should_stop = False
        else:
            # 其他情况，保持当前轨迹
            print("保持轨迹不变")
            pass

