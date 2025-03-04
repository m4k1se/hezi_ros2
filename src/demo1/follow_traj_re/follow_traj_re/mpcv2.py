import os
import sys
import time
import math
import threading
import csv
import numpy as np
import matplotlib.pyplot as plt
import cvxpy
import can
from math import radians, cos, sin, asin, sqrt, degrees, atan2
from pynput import keyboard
from pyproj import CRS, Transformer


# 车辆参数
VEHICLE_WIDTH = 1.9   # m
VEHICLE_LENGTH = 4.5  # m
WHEEL_FACTOR = 7.2
manual_triggered = False
stop_record = False
mod_666 = 0
mod_AE = 0

# MPC参数
NX = 4  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]
T = 3  # horizon length

R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 1.0])  # input difference cost matrix
Q = np.diag([1.0, 1.0, 0.5, 0.5])  # state cost matrix
Qf = Q  # state final matrix
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 500.0  # max simulation time

MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish param

TARGET_SPEED = 10.0 / 3.6  # [m/s] target speed
N_IND_SEARCH = 100  # Search index number

DT = 0.5  # [s] time tick

# Vehicle parameters
LENGTH = 4.5  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 2.5  # [m]

MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_SPEED = 55.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/s^2]

show_animation = True



# 初始化 WGS84 到 UTM 的投影转换器
wgs84 = CRS("EPSG:4326")
utm_zone_number = 50  # 根据实际情况选择合适的 UTM 区域
utm_crs = CRS(f"EPSG:{32600 + utm_zone_number}")  # 例如，UTM Zone 50N 对应 EPSG:32650
projector_to_utm = Transformer.from_crs(wgs84, utm_crs, always_xy=True)
projector_to_wgs84 = Transformer.from_crs(utm_crs, wgs84, always_xy=True)

def latlon_to_utm(lon, lat):
    """将经纬度转换为 UTM 坐标"""
    x, y = projector_to_utm.transform(lon, lat)
    return x, y

def smooth_yaw_iter(previous_yaw, new_yaw):
    """
    Smooth the yaw angle based on the previous yaw to ensure continuity.

    :param previous_yaw: (float) Previous yaw angle in radians
    :param new_yaw: (float) New yaw angle in radians (not yet normalized)
    :return: (float) Smoothed and normalized yaw angle in radians within (-pi, pi]
    """
    dyaw = new_yaw - previous_yaw

    # 调整 dyaw，使其在 [-pi, pi] 范围内
    dyaw = (dyaw + np.pi) % (2.0 * np.pi) - np.pi

    # 平滑后的 yaw
    smoothed_yaw = previous_yaw + dyaw

    return smoothed_yaw

# ======================= MPC 控制逻辑 =========================

class State:
    """
    Vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None
        self.oa = None
        self.oldelta = None

def pi_2_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def get_linear_model_matrix(v, phi, delta):
    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = DT * math.cos(phi)
    A[0, 3] = - DT * v * math.sin(phi)
    A[1, 2] = DT * math.sin(phi)
    A[1, 3] = DT * v * math.cos(phi)
    A[3, 2] = DT * math.tan(delta) / WB

    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)

    C = np.zeros(NX)
    C[0] = DT * v * math.sin(phi) * phi
    C[1] = - DT * v * math.cos(phi) * phi
    C[3] = - DT * v * delta / (WB * math.cos(delta) ** 2)

    return A, B, C

def get_nparray_from_matrix(x):
    return np.array(x).flatten()

def linear_mpc_control(xref, xbar, x0, dref):
    """
    Linear MPC control
    """
    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)  # input cost

        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)  # state cost

        A, B, C = get_linear_model_matrix(
            xbar[2, t], xbar[3, t], dref[0, t])
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]  # dynamics constraints

        if t < (T - 1):
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)  # input difference cost
            constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= 
                            MAX_DSTEER * DT]

    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)  # final state cost

    constraints += [x[:, 0] == x0]
    constraints += [x[2, :] <= MAX_SPEED]
    constraints += [x[2, :] >= MIN_SPEED]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.CLARABEL, verbose=False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        ox = get_nparray_from_matrix(x.value[0, :])
        oy = get_nparray_from_matrix(x.value[1, :])
        ov = get_nparray_from_matrix(x.value[2, :])
        oyaw = get_nparray_from_matrix(x.value[3, :])
        oa = get_nparray_from_matrix(u.value[0, :])
        odelta = get_nparray_from_matrix(u.value[1, :])
    else:
        print("Error: Cannot solve mpc..")
        oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

    return oa, odelta, ox, oy, oyaw, ov

def predict_motion(x0, oa, od, xref):
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
    for (ai, di, i) in zip(oa, od, range(1, T + 1)):
        state = update_state(state, ai, di)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.v
        xbar[3, i] = state.yaw

    return xbar

def iterative_linear_mpc_control(xref, x0, dref, oa, od):
    """
    MPC control with updating operational point iteratively
    """
    ox, oy, oyaw, ov = None, None, None, None

    if oa is None or od is None:
        oa = [0.0] * T  # Initialize acceleration sequence
        od = [0.0] * T  # Initialize steering angle sequence

    for i in range(MAX_ITER):
        xbar = predict_motion(x0, oa, od, xref)
        poa, pod = oa[:], od[:]
        oa, odelta, ox, oy, oyaw, ov = linear_mpc_control(xref, xbar, x0, dref)
        du = sum(abs(oa - poa)) + sum(abs(odelta - pod))  # Change in control
        if du <= DU_TH:  # Check convergence
            break
    else:
        print("Iterative is max iter")

    return oa, odelta, ox, oy, oyaw, ov

def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, pind):
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)

    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)

    if pind >= ind:
        ind = pind

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = sp[ind]
    xref[3, 0] = cyaw[ind]
    dref[0, 0] = 0.0  # steer operational point should be 0

    travel = 0.0

    for i in range(T + 1):
        travel += abs(state.v) * DT  # cumulative distance
        dind = int(round(travel / dl))  # number of path points traveled

        if (ind + i) < ncourse:
            xref[0, i] = cx[ind + i]
            xref[1, i] = cy[ind + i]
            xref[2, i] = sp[ind + i]
            xref[3, i] = cyaw[ind + i]
            dref[0, i] = 0.0
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = sp[ncourse - 1]
            xref[3, i] = cyaw[ncourse - 1]
            dref[0, i] = 0.0

    return xref, ind, dref

def check_goal(state, goal, tind, nind):
    # Check if goal is reached
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)

    isgoal = (d <= GOAL_DIS)

    if abs(tind - nind) >= 5:
        isgoal = False

    isstop = (abs(state.v) <= STOP_SPEED)

    if isgoal and isstop:
        return True

    return False

def update_state(state, a, delta):
    # Input check
    if delta >= MAX_STEER:
        delta = MAX_STEER
    elif delta <= -MAX_STEER:
        delta = -MAX_STEER

    state.x = state.x + state.v * math.cos(state.yaw) * DT
    state.y = state.y + state.v * math.sin(state.yaw) * DT
    state.yaw = state.yaw + state.v / WB * math.tan(delta) * DT
    state.yaw = pi_2_pi(state.yaw)
    state.v = state.v + a * DT

    if state.v > MAX_SPEED:
        state.v = MAX_SPEED
    elif state.v < MIN_SPEED:
        state.v = MIN_SPEED

    return state

def calc_nearest_index(state, cx, cy, cyaw, pind):
    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)
    ind = d.index(mind) + pind
    mind = math.sqrt(mind)  # minimum distance

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind

# ======================= 车辆循迹控制逻辑 =========================

class ISGSpeedFilter:
    def __init__(self):
        self.isg_sum_filtspd = 0  # 总和
        self.isg_mot_spd_filt = 0  # 滤波后的速度
        self.isg_mot_spd = 0  # 实时速度
        self.MAT_Moto_spd = 0  # 最终输出的速度

    def update_speed(self, isg_mot_spd):
        self.isg_mot_spd = isg_mot_spd
        self.isg_sum_filtspd += self.isg_mot_spd  # 加上当前速度
        self.isg_sum_filtspd -= self.isg_mot_spd_filt  # 减去上一次的滤波结果
        
        # 计算滤波后的速度
        self.isg_mot_spd_filt = self.isg_sum_filtspd / 15
        self.MAT_Moto_spd = self.isg_mot_spd_filt  # 更新最终输出速度

        return self.MAT_Moto_spd

class VehicleTrajectoryFollower:
    def __init__(self, trajectory_csv, target_index=30):
        """
        初始化，读取轨迹点
        :param trajectory_csv: 包含轨迹点的CSV文件路径，轨迹点格式为[经度, 纬度, 航向角度, 速度]
        """
        self.cx, self.cy, self.cyaw, self.ck, self.sp = self.read_reference_trajectory(trajectory_csv)
        self.closest_index = 0
        self.offset_target_index = target_index
        self.wheelbase = 2.85
        self.target_index = 0
        
        self.previous_turn_angle = 0
        self.max_turn_rate = 6  # 限制每次转向角的最大变化速率（度）
        
        # 平滑航向角
        self.cyaw = smooth_yaw(self.cyaw)
        
    def upate_target_index(self, current_position):
        dx = [current_position[0] - icx for icx in self.cx]
        dy = [current_position[1] - icy for icy in self.cy]
        
        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        
        self.closest_index = d.index(min(d))

    def read_reference_trajectory(self, csv_file_path):
        # 使用MPC的读取函数读取轨迹，包括速度
        cx, cy, cyaw, ck = read_csv(csv_file_path)
        sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)
        return cx, cy, cyaw, ck, sp

    def smooth_turn_angle(self, turn_angle):
        # 限制转向角的最大变化速率
        angle_difference = turn_angle - self.previous_turn_angle
        if angle_difference > self.max_turn_rate:
            update_turn_angle = self.previous_turn_angle + self.max_turn_rate
        elif angle_difference < -self.max_turn_rate:
            update_turn_angle = self.previous_turn_angle - self.max_turn_rate
        else:
            update_turn_angle = turn_angle
        print(f"input:{turn_angle}======>update:{update_turn_angle}")
        # 更新上一次的转向角
        self.previous_turn_angle = update_turn_angle
        return update_turn_angle

    def calculate_turn_angle(self, current_position, current_heading, current_speed, mpc_controller):
        current_x, current_y, _ = current_position
        
        # 计算当前位置到轨迹上最近点的距离和索引
        # TODO: 可以设计滑动窗口，减少计算量
        closest_index, _ = calc_nearest_index(mpc_controller, self.cx, self.cy, self.cyaw, 0)
        
        # 准备MPC的参考轨迹
        xref, _, dref = calc_ref_trajectory(
            State(x=current_x, y=current_y, v=current_speed, yaw=current_heading),
            self.cx, self.cy, self.cyaw, self.ck, self.sp, 1.0, closest_index
        )

        # 当前状态
        x0 = [current_x, current_y, current_speed, current_heading]

        # 使用MPC控制器计算控制输入
        oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(
            xref, x0, dref, mpc_controller.oa, mpc_controller.odelta
        )

        di, ai = 0.0, 0.0
        if odelta is not None:
            di, ai = odelta[0], oa[0]
            di = math.degrees(di)
            print(f"MPC Output - di: {di}, ai: {ai}")
            # 限制转向角
            di = max(min(di, math.degrees(MAX_STEER)), math.degrees(-MAX_STEER))
        
        # 将转向角进行平滑处理
        filtered_angle = self.smooth_turn_angle(di*7.2)
        
        # 更新MPC控制器的上一轮控制输入
        mpc_controller.oa = oa
        mpc_controller.odelta = odelta

        return filtered_angle, ai

# ======================= CAN 总线通信逻辑 =========================

class CanUse:
    def __init__(self, zone):
        self.bus_ins = can.interface.Bus(channel='can0', bustype='socketcan')
        self.bus_vcu = can.interface.Bus(channel='can1', bustype='socketcan')
        self.ego_lon = 31.8925019
        self.ego_lat = 118.8171577
        self.ego_yaw_rad = 0
        self.ego_v = 3
        self.ego_a = 0
        self.eps_mode = 2
        self.auto_driver_allowed = False
        self.ego_x, self.ego_y = latlon_to_utm(self.ego_lon, self.ego_lat)
        
        # 用于平滑航向角
        self.previous_yaw = math.radians(self.ego_yaw_rad)

    def read_ins_info(self):
        """获取惯导的主车信息"""
        message_ins = self.bus_ins.recv(timeout=0.1)
        message_vcu = self.bus_vcu.recv(timeout=0.1)
        if message_ins is not None and message_ins.arbitration_id == 0x504:
            # 解析位置数据
            can_data = message_ins.data
            INS_Latitude = (can_data[0] << 24) | (can_data[1] << 16) | (can_data[2] << 8) | can_data[3]
            INS_Longitude = (can_data[4] << 24) | (can_data[5] << 16) | (can_data[6] << 8) | can_data[7]
            INS_Latitude = INS_Latitude * 0.0000001 - 180
            INS_Longitude = INS_Longitude * 0.0000001 - 180
 
            
            # 将经纬度转换为 UTM 坐标
            ego_x, ego_y = latlon_to_utm(INS_Longitude, INS_Latitude)
            self.ego_x = ego_x
            self.ego_y = ego_y
            
             
        if message_ins is not None and message_ins.arbitration_id == 0x505:
            speed_data = message_ins.data
            INS_NorthSpd =  (speed_data[0] << 8) | speed_data[1]
            INS_NorthSpd =   INS_NorthSpd * 0.0030517 - 100    # m/s
            INS_NorthSpd *= 3.6
            INS_EastSpd =  (speed_data[2] << 8) | speed_data[3]
            INS_EastSpd =   INS_EastSpd * 0.0030517 - 100    # m/s
            INS_EastSpd *= 3.6
            INS_ToGroundSpd =  (speed_data[4] << 8) | speed_data[5]
            INS_ToGroundSpd =   INS_ToGroundSpd * 0.0030517 - 100    # m/s
            INS_ToGroundSpd *= 3.6
            speed =  sqrt(INS_EastSpd**2 + INS_NorthSpd**2 + INS_ToGroundSpd**2)
            self.ego_v = speed

        if message_ins is not None and message_ins.arbitration_id == 0x502:
            Angle_data = message_ins.data
            HeadingAngle =  (Angle_data[4] << 8) | Angle_data[5]
            HeadingAngle =   HeadingAngle * 0.010986 - 360
            # self.ego_yaw = HeadingAngle 
            
            # 将航向角从 INS 坐标系转换为 UTM 坐标系
            # INS: 0° 正北，东为正
            # UTM: 0° 正东，北为正
            # 转换公式：UTM_yaw = 90 - INS_yaw
            utm_yaw_deg = 90 - HeadingAngle
            utm_yaw_rad = math.radians(utm_yaw_deg)

            # 平滑航向角
            smoothed_yaw = smooth_yaw_iter(self.previous_yaw, utm_yaw_rad)
            self.previous_yaw = smoothed_yaw
            self.ego_yaw = smoothed_yaw
            self.ego_yaw_deg = math.degrees(smoothed_yaw)  # 转换回度数用于其他部分
            # print(f"Smoothed Yaw (deg): {self.ego_yaw}")
            
        if message_ins is not None and message_ins.arbitration_id == 0x500:
            acc_data = message_ins.data
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
            data3 = int(250 / 10) & 0xFF         # data3缩放到8位范围，0-255, angle_spd=100
            data4 = int(enable) & 0x01          # data4缩放到1位范围，0或1
                
            # 构建发送数据，确保8字节长度
            data = [data1_high, data1_low, data2, data3, data4, 0, 0, 0]

            # 创建CAN消息，ID设置为0x0AE
            msg = can.Message(arbitration_id=id, data=data, is_extended_id=False)
            self.bus_vcu.send(msg)
        
        if action_type == "acc":
            auto_drive_cmd_bits = mod & 0x07  # 取最低3位
            speed_scaled = int(action[0]) & 0x1F  # 取5位（位3-7）
            byte0 = (speed_scaled << 3) | auto_drive_cmd_bits

            angle_scaled = int((action[1] - (-500)) / 0.1) & 0xFFFF  # 16位
            byte1 = (angle_scaled >> 8) & 0xFF  # 高8位
            byte2 = angle_scaled & 0xFF         # 低8位

            acc = action[2]
            acc_scaled = int((acc - (-4)) / 1) & 0xFF  # 假设缩放因子1，偏移量-4

            # 构建发送数据，剩余字节填充0
            data_666 = [byte0, byte1, byte2, acc_scaled, 0, 0, 0, 0]
            
            msg = can.Message(arbitration_id=id, data=data_666, is_extended_id=False)
            self.bus_vcu.send(msg)

# ======================= 辅助函数 =========================

def read_csv(csv_file_path): 
    x_coords = []
    y_coords = []
    heading_list = []
    speed_list = []
    ck = []  # 假设CSV中有曲率信息，如果没有，请相应调整
    # 打开CSV文件并读取内容  
    with open(csv_file_path, mode='r', newline='') as file:  
        csv_reader = csv.reader(file)  
        
        # 跳过标题行（如果有的话）  
        headers = next(csv_reader, None)  # 这行代码会读取第一行，如果第一行是标题则跳过  
        
        # 读取每一行数据并添加到列表中  
        for row in csv_reader:  
            lon = float(row[0])  
            lat = float(row[1])
            heading = float(row[2])
            speed = float(row[3])
            curvature = float(row[4]) if len(row) > 4 else 0.0  # 假设曲率在第五列
            # 将经纬度转换为UTM坐标
            x_coords.append(lon)
            y_coords.append(lat)
            heading_list.append(math.radians(heading))  # 转换为弧度
            speed_list.append(speed)
            ck.append(curvature)
            # 将UTM坐标和航向角存储到traj_data中  
    return x_coords, y_coords, heading_list, ck

def calc_speed_profile(cx, cy, cyaw, target_speed):
    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

    speed_profile[-1] = 0.0

    return speed_profile

def smooth_yaw(yaw):
    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw

# ======================= 可视化函数 =========================

def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover
    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")

# ======================= 键盘监听逻辑 =========================

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

# ======================= 主控制逻辑 =========================

def start_main(shared_data, can_use, follower, speed_filter, mpc_controller):
    global mod_666
    global mod_AE
    global manual_triggered
    can_use.read_ins_info()
    follower.upate_target_index((can_use.ego_x, can_use.ego_y))

    while True:
        can_use.read_ins_info()
        if can_use.eps_mode != 3 and manual_triggered:
            mod_AE = 1
            mod_666 = 1
        if can_use.eps_mode == 3:
            mod_AE = 3
            mod_666 = 0
            manual_triggered = False
        if mod_AE == 1 and mod_666 == 1:
            if can_use.ego_x is not None and can_use.ego_y is not None:
                turn_angle, accel = follower.calculate_turn_angle(
                    (can_use.ego_x, can_use.ego_y, can_use.ego_yaw),
                    can_use.ego_yaw,
                    can_use.ego_v,
                    mpc_controller
                )

                filtered_angle = speed_filter.update_speed(turn_angle)
                print("filtered angle", filtered_angle)
                new_frame = [5, filtered_angle, 0]
            else:
                print("主车定位丢失...")
                new_frame = [0, 0, 0]
        else:
            print("请按's'进入自动驾驶模式...")
            new_frame = [0, 0, 0]
            continue
        with shared_data['lock']:
            shared_data['frame'] = new_frame

# ======================= 发送CAN帧逻辑 =========================

def send_frame(shared_data, can_use):
    last_frame = None
    while True:
        # 每隔0.005秒发送一帧（200帧每秒）
        time.sleep(0.005)
        with shared_data["lock"]:
            if shared_data["frame"] is not None:
                last_frame = shared_data["frame"]

        if last_frame is not None:
            can_use.publish_planner_action(
                action=last_frame,
                id=0x666,
                action_type="acc",
                mod=1,
                enable=1
            )

# ======================= 仿真逻辑（可选） =========================

def do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state):
    """
    Simulation

    cx: course x position list
    cy: course y position list
    cyaw: course yaw position list
    ck: course curvature list
    sp: speed profile
    dl: course tick [m]
    initial_state: initial state of the vehicle
    """

    goal = [cx[-1], cy[-1]]

    state = initial_state

    # initial yaw compensation
    if state.yaw - cyaw[0] >= math.pi:
        state.yaw -= math.pi * 2.0
    elif state.yaw - cyaw[0] <= -math.pi:
        state.yaw += math.pi * 2.0

    time_sim = 0.0
    x = [state.x]
    y = [state.y]
    yaw_sim = [state.yaw]
    v = [state.v]
    t = [0.0]
    d = [0.0]
    a = [0.0]
    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)
    print(target_ind)
    odelta, oa = None, None

    cyaw = smooth_yaw(cyaw)

    while MAX_TIME >= time_sim:
        xref, target_ind, dref = calc_ref_trajectory(
            state, cx, cy, cyaw, ck, sp, dl, target_ind)
    
        x0 = [state.x, state.y, state.v, state.yaw]  # current state

        oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(
            xref, x0, dref, oa, odelta)
        di, ai = 0.0, 0.0
        if odelta is not None:
            di, ai = odelta[0], oa[0]
            print(f"di:{di}, ai:{ai}")
            # di 大于max，就取max，小于min，就取min
            if di >= MAX_STEER:
                di = MAX_STEER
            elif di <= -MAX_STEER:
                di = -MAX_STEER
            state = update_state(state, ai, di)

        time_sim = time_sim + DT

        x.append(state.x)
        y.append(state.y)
        yaw_sim.append(state.yaw)
        v.append(state.v)
        t.append(time_sim)
        d.append(di)
        a.append(ai)

        if check_goal(state, goal, target_ind, len(cx)):
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            if ox is not None:
                plt.plot(ox, oy, "xr", label="MPC")
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plot_car(state.x, state.y, state.yaw, steer=di)
            plt.axis("equal")
            plt.grid(True)
            plt.title("Time[s]:" + str(round(time_sim, 2))
                      + ", speed[km/h]:" + str(round(state.v * 3.6, 2)))
            plt.pause(0.0001)

    return t, x, y, yaw_sim, v, d, a

# ======================= 主函数 =========================

def main():
    # 初始化
    trajectory_csv = '/home/nvidia/vcii/follow_trajectory/collect_trajectory/processed_straight12_17_with_yaw_ck.csv'
    follower = VehicleTrajectoryFollower(trajectory_csv=trajectory_csv, target_index=2)
    speed_filter = ISGSpeedFilter()
    can_use = CanUse(zone=49)
    
    # 初始化MPC控制器状态
    mpc_controller = State()
    mpc_controller.oa = None
    mpc_controller.odelta = None

    # 用于在线程之间共享数据
    shared_data = {
        "frame": None,  # 存储最新的帧
        "lock": threading.Lock()  # 用于保证线程安全的锁
    }
    
    # 创建线程
    compute_thread = threading.Thread(target=start_main, args=(shared_data, can_use, follower, speed_filter, mpc_controller))
    send_thread = threading.Thread(target=send_frame, args=(shared_data, can_use))
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)

    # 启动线程
    keyboard_thread.start()
    compute_thread.start()
    send_thread.start()
        
    # 主线程等待计算和发送线程完成（通常不会退出）
    compute_thread.join()
    send_thread.join()

if __name__ == '__main__':
    main()
