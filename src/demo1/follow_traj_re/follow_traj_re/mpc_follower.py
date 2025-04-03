"""

Path tracking simulation with iterative linear model predictive control for speed and steer control

author: Atsushi Sakai (@Atsushi_twi)

"""
import matplotlib.pyplot as plt
import time
import cvxpy
import math
import numpy as np
import sys
import pathlib
import csv
from pyproj import Proj
from dataclasses import dataclass
import pandas as pd
from lane_change import LaneChangeDecider
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

# from utils.angle import angle_mod

WHEEL_FACTOR = 7.2

NX = 4  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]
T = 30  # horizon length

# mpc parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 1.0])  # input difference cost matrix
Q = np.diag([1.0, 1.0, 0.5, 0.5])  # state cost matrix
Qf = Q  # state final matrix
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 500.0  # max simulation time

# iterative paramter
MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish param

TARGET_SPEED = 10.0 / 3.6  # [m/s] target speed
N_IND_SEARCH = 10  # Search index number

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
MAX_ACCEL = 1.0  # maximum accel [m/ss]

show_animation = True

def pi_2_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def get_nparray_from_matrix(x):
    return np.array(x).flatten()

from pyproj import Proj

lonlat2xy_old = Proj('+proj=tmerc +lon_0=118.8170043 +lat_0=31.8926311 +ellps=WGS84')
def read_csv(csv_file_path): 
    x_coords = []
    y_coords = []
    heading_list = []
    speed_list = []
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
            # 将经纬度转换为UTM坐标
            x_coords.append(lon)
            y_coords.append(lat)
            heading_list.append(heading)
            speed_list.append(float(row[3]))
            # 将UTM坐标和航向角存储到traj_data中  
    return x_coords, y_coords, heading_list, speed_list

class Simulator:
    
    def __init__(self, initial_state, follower):
        self.state = initial_state
        self.x = [self.state.x]
        self.y = [self.state.y]
        self.yaw = [self.state.yaw]
        self.v = [self.state.v]
        self.follower = follower
    
    def update_state(self, a, delta):
        self.state.x = self.state.x + self.state.v * math.cos(self.state.yaw) * DT
        self.state.y = self.state.y + self.state.v * math.sin(self.state.yaw) * DT
        self.state.yaw = self.state.yaw + self.state.v / WB * math.tan(delta) * DT
        # state.yaw = normalize_angle(state.yaw)
        self.state.v = self.state.v + a * DT

        if self.state.v > MAX_SPEED:
            self.state.v = MAX_SPEED
        elif self.state.v < MIN_SPEED:
            self.state.v = MIN_SPEED
            
    def plot_car(self, x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

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
    
    def do_simulation(self, ai, di, cx, cy, cyaw, target_ind):
        
        self.update_state(ai, di)
        self.x.append(self.state.x)
        self.y.append(self.state.y)
        self.yaw.append(self.state.yaw)
        self.v.append(self.state.v)

        if show_animation:  # pragma: no cover
            # plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(cx, cy, "-r", label="course", alpha=0.5)
            plt.plot(self.x, self.y, "ob", label="trajectory")
            # plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            self.plot_car(self.state.x, self.state.y, self.state.yaw, steer=di)
            plt.axis("equal")
            plt.grid(True)
            plt.title("Time[s]:" + str(round(0, 2))
                    + ", speed[km/h]:" + str(round(self.state.v * 3.6, 2)))
            plt.legend()
            plt.pause(0.0001)
                    
class ObsItem:
    def __init__(self, x=0, y=0, v=0, yaw=0, width=0, length=0):
        self.x = x
        self.y = y
        self.v = v
        self.yaw = yaw
        self.width = width
        self.length = length

class ObsPublish:
    def __init__(self):
        # 每个障碍物以 ObsItem 形式存储
        self.obs_dict = {
            "cone": {
                "12345": ObsItem(x=0.0, 
                                 y=9.0, 
                                 v=0, 
                                 yaw=1.68, 
                                 width=2, 
                                 length=2),
            },
            "pedestrain": {},
            "vehicle": {},
            "bycicle": {}
        }

    def __iter__(self):
        return iter(self.obs_dict)

    def __getitem__(self, key):
        return self.obs_dict[key]

class MPCfollower:
    def __init__(self, path):
        self.cx, self.cy, self.cyaw, self.ck = read_csv(path)
        self.sp = self.calc_speed_profile(TARGET_SPEED)
        self.cyaw = self.smooth_yaw(self.cyaw)
        self.oa = None
        self.odelta = None
        self.ai = 0
        self.di = 0
        self.target_ind = 0
        self.previous_turn_angle = 0
        self.max_turn_rate = 6
        self.all_line_temp = []
        self.ref_line = []
        self.obs = []
        self.planning = False
        self.init_mpc()
        self.decider = LaneChangeDecider(path)
        self.decider.init_refline()
    
    def init_mpc(self):
        self.goal = [self.cx[-1], self.cy[-1]]
        self.state = State(x=self.cx[0], y=self.cy[0], yaw=self.cyaw[0], v=0.0)
        # initial yaw compensation
        if self.state.yaw - self.cyaw[0] >= math.pi:
            self.state.yaw -= math.pi * 2.0
        elif self.state.yaw - self.cyaw[0] <= -math.pi:
            self.state.yaw += math.pi * 2.0
        self.target_ind, _ = self.calc_nearest_index(self.state, 0)
        self.odelta, self.oa = None, None
        self.cyaw = self.smooth_yaw(self.cyaw)
        for i in range(len(self.cx)):
            self.all_line_temp.append([
                self.cx[i],
                self.cy[i],
                self.cyaw[i],
                self.ck[i],
                self.sp[i]
            ])

    def update_target_index(self, state):
        self.state.x = state.x
        self.state.y = state.y
        # self.state.yaw = math.radians(ego_yaw)
        self.state.yaw = state.yaw
        self.state.v = state.v
        self.state.predelta = 0
        self.target_ind, _ = self.calc_nearest_index(self.state, 0)
        print("self target indx",  self.target_ind)
    
    def smooth_yaw(self, yaw):
        for i in range(len(yaw) - 1):
            dyaw = yaw[i + 1] - yaw[i]
            while dyaw >= math.pi / 2.0:
                yaw[i + 1] -= math.pi * 2.0
                dyaw = yaw[i + 1] - yaw[i]
            while dyaw <= -math.pi / 2.0:
                yaw[i + 1] += math.pi * 2.0
                dyaw = yaw[i + 1] - yaw[i]
        return yaw
    
    def normalize_angle_rad(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle
        
    def calc_speed_profile(self, target_speed):
        speed_profile = [target_speed] * len(self.cx)
        direction = 1.0  # forward
        # Set stop point
        for i in range(len(self.cx) - 1):
            dx = self.cx[i + 1] - self.cx[i]
            dy = self.cy[i + 1] - self.cy[i]
            move_direction = math.atan2(dy, dx)
            if dx != 0.0 and dy != 0.0:
                dangle = abs(pi_2_pi(move_direction - self.cyaw[i]))
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
    
    def get_linear_model_matrix(self, v, phi, delta):
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
    
    def update_state(self, state, a, delta):
        # input check
        if delta >= MAX_STEER:
            delta = MAX_STEER
        elif delta <= -MAX_STEER:
            delta = -MAX_STEER
        
        state.x = state.x + state.v * math.cos(state.yaw) * DT
        state.y = state.y + state.v * math.sin(state.yaw) * DT
        state.yaw = state.yaw + state.v / WB * math.tan(delta) * DT
        # state.yaw = normalize_angle(state.yaw)
        state.v = state.v + a * DT

        if state.v > MAX_SPEED:
            state.v = MAX_SPEED
        elif state.v < MIN_SPEED:
            state.v = MIN_SPEED
        return state
    
    def calc_nearest_index(self, state, pind):
        # print(len(cx[pind:(pind + N_IND_SEARCH)]))
        dx = [state.x - icx for icx in self.cx[pind:(pind + N_IND_SEARCH)]]
        dy = [state.y - icy for icy in self.cy[pind:(pind + N_IND_SEARCH)]]

        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

        mind = min(d)

        ind = d.index(mind) + pind

        mind = math.sqrt(mind) # 最小距离

        dxl = self.cx[ind] - state.x
        dyl = self.cy[ind] - state.y

        angle = pi_2_pi(self.cyaw[ind] - math.atan2(dyl, dxl))
        if angle < 0:
            mind *= -1
        
        return ind, mind
    
    def calc_ref_trajectory(self, state, dl, pind):
        xref = np.zeros((NX, T + 1))
        dref = np.zeros((NU, T + 1))
        ncourse = len(self.cx)
        temp_ind, _ = self.calc_nearest_index(state, pind)
        ind = temp_ind
        # print("最近点的索引：", ind, pind)
        if pind >= ind:
            ind = pind
        xref[0, 0] = self.cx[ind]
        xref[1, 0] = self.cy[ind]
        xref[2, 0] = self.sp[ind]
        xref[3, 0] = self.cyaw[ind]
        dref[1, 0] = math.atan2(WB * self.ck[ind], 1.0)  # 1.0 is just L
        
        print("xref: "  , xref[0, 0], xref[1, 0], xref[2, 0], xref[3, 0])
        print("state: " , state.x, state.y, state.v, state.yaw)
        travel = 0.0

        for i in range(T + 1):
            travel += abs(state.v) * DT  # 累计形式的距离
            dind = int(round(travel / dl))  # dl是路径点的间隔，travel/dl是当前车辆已经行驶的路径点数

            if (ind + dind) < ncourse:  #n course是路径点的总数，判断是否超出路径点的总数
                # xref[0, i] = cx[ind + dind]
                # xref[1, i] = cy[ind + dind]
                # xref[2, i] = sp[ind + dind]
                # xref[3, i] = cyaw[ind + dind]
                # dref[0, i] = 0.0
            # if (ind + i) < ncourse:
                xref[0, i] = self.cx[ind + i]
                xref[1, i] = self.cy[ind + i]
                xref[2, i] = self.sp[ind + i]
                xref[3, i] = self.cyaw[ind + i]
                dref[1, i] = math.atan2(WB * self.ck[ind + i], 1.0)
            else:
                xref[0, i] = self.cx[ncourse - 1]
                xref[1, i] = self.cy[ncourse - 1]
                xref[2, i] = self.sp[ncourse - 1]
                xref[3, i] = self.cyaw[ncourse - 1]
                dref[1, i] = math.atan2(WB * self.ck[ncourse - 1], 1.0)

        return xref, ind, dref
    
    def predict_motion(self, x0, oa, od, xref):
        xbar = xref * 0.0
        for i, _ in enumerate(x0):
            xbar[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, T + 1)):
            state = self.update_state(state, ai, di)
            xbar[0, i] = state.x
            xbar[1, i] = state.y
            xbar[2, i] = state.v
            xbar[3, i] = state.yaw

        return xbar

    def iterative_linear_mpc_control(self, xref, x0, dref, oa, od):
        ox, oy, oyaw, ov = None, None, None, None

        if oa is None or od is None:
            oa = [0.0] * T  # 上一轮优化得到的加速度序列，如果是第一次迭代，就初始化为0
            od = [0.0] * T  # 上一轮优化得到的转角序列，如果是第一次迭代，就初始化为0

        for i in range(MAX_ITER):
            xbar = self.predict_motion(x0, oa, od, xref)
            poa, pod = oa[:], od[:]
            oa, od, ox, oy, oyaw, ov = self.linear_mpc_control(xref, xbar, x0, dref)
            du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value 用于判断是否收敛
            if du <= DU_TH: # 如果u的变化小于阈值，说明收敛了, 就退出迭代
                break
        else:
            print("Iterative is max iter")

        return oa, od, ox, oy, oyaw, ov

    def linear_mpc_control(self, xref, xbar, x0, dref):
        x = cvxpy.Variable((NX, T + 1))
        u = cvxpy.Variable((NU, T))

        cost = 0.0
        constraints = []

        for t in range(T):
            cost += cvxpy.quad_form(u[:, t], R)  # cost function 控制输入的cost

            if t != 0:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)  # state cost function

            A, B, C = self.get_linear_model_matrix(
                xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C] # 动力学约束 保证车辆的运动符合动力学模型

            if t < (T - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)  # input difference cost function 控制输入的变化cost，目的是让控制输入尽量平滑，减小震荡
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <=  # 控制输入的变化率约束，限制转角的变化率，防止转角变化过快
                                MAX_DSTEER * DT]

        cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf) # final state cost function 终端状态的cost

        constraints += [x[:, 0] == x0]  
        constraints += [x[2, :] <= MAX_SPEED]  # 速度约束
        constraints += [x[2, :] >= MIN_SPEED]
        constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL] # 加速度约束
        constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.CLARABEL, verbose=False)
        # prob.solve(solver=cvxpy.ECOS, verbose=False)
        
        
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
    
    def smooth_turn_angle(self, turn_angle):
        angle_diff = turn_angle - self.previous_turn_angle
        if angle_diff > self.max_turn_rate:
            update_turn_angle = self.previous_turn_angle + self.max_turn_rate
        elif angle_diff < -self.max_turn_rate:
            update_turn_angle = self.previous_turn_angle - self.max_turn_rate
        else:
            update_turn_angle = turn_angle
        print(f"input:{turn_angle}======>update:{update_turn_angle}")
        self.previous_turn_angle = update_turn_angle
        return update_turn_angle
    
    def cal_acc_and_delta(self, state):
        def convert_angle(angle):
            if angle < 0:
                angle_2 = angle + 360
                if abs(angle) > abs(angle_2):
                    return angle_2
                else:
                    return angle
            else:
                angle_2 = angle-360
                if abs(angle) >abs(angle_2):
                    return angle_2
                else:
                    return angle
        if state.yaw - self.cyaw[0] > math.pi:
            state.yaw -= math.pi * 2.0
        elif state.yaw - self.cyaw[0] < -math.pi:
            state.yaw += math.pi * 2.0
        xref, self.target_ind, dref = self.calc_ref_trajectory(state, 1.0, self.target_ind)
        x0 = [state.x, state.y, state.v, state.yaw]
        self.oa, self.odelta, ox, oy, oyaw, ov = self.iterative_linear_mpc_control(
            xref, x0, dref, self.oa, self.odelta)
        
        print("x0:         ", x0)
        print("ox:         ", ox)
        print("x reference:", xref[0])
        print("oy: ", oy)
        print("y reference:", xref[1])
        print("oyaw: ", oyaw)
        print("yaw reference:", xref[3])
        print("ov: ", ov)
        print("v reference:", xref[2])
        print("o delt",self.odelta)
        print("delta reference:", dref[1])
        
        plt.plot(ox, oy, c='r')
        plt.plot(self.cx, self.cy, c='g')
        plt.plot(xref[0],xref[1],c='b')
        plt.axis("equal")
        plt.scatter(self.state.x, self.state.y, c='y')
        plt.scatter(self.cx[self.target_ind], self.cy[self.target_ind], c='b')
        # plt.savefig("prediction_points_bank.png")
        
        if self.odelta is not None:
            self.di, self.ai = self.odelta[0], self.oa[0]
            self.di = self.normalize_angle_rad(self.di)
            print(f"MPC Output - di: {self.di}, ai: {self.ai}")
            self.di = max((min(self.di, MAX_STEER)), -MAX_STEER)
                        
            # di_deg = math.degrees(self.di)
            # di_deg = convert_angle(di_deg)
            # if di_deg*WHEEL_FACTOR > 460:
            #     turn_angle = 460
            # elif di_deg*WHEEL_FACTOR < -460:
            #     turn_angle = -460
            # else: 
            #     turn_angle = di_deg * WHEEL_FACTOR
            # self.di = self.smooth_turn_angle(-turn_angle)
            
            # print(f"Fielterd Output - di:{self.di}, ai:{self.ai}")
    
        return self.ai, self.di
     
    def act(self, state, obs_publish: ObsPublish):
        plt.cla()
        self.decider.update_state(state, obs_publish)
        self.cx, self.cy, self.cyaw, self.ck, self.sp = self.decider.publish_new_refline()
        # self.GenerateLaneBorrow(data_cone, state)
        # self.sp = self.calc_speed_profile(TARGET_SPEED)
        ai, di = self.cal_acc_and_delta(state)
        return ai, di
        
class State:
    """
    vehicle state class
    """
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None

def main():
    print(__file__ + " start!!")
    start = time.time()
    
    mpc = MPCfollower(r'/home/renth/follow_trajectory/collect_trajectory/processed_straight12_17_with_yaw_ck.csv')
    obs_publish = ObsPublish()
    start_x = mpc.cx[0]
    start_y = mpc.cy[0]
    start_yaw = mpc.cyaw[0]
    initial_state = State(x=start_x, y=start_y, yaw=start_yaw, v=0.0)
    sim = Simulator(initial_state, mpc)
    
    for i in range(500):
        ai, di = mpc.act(sim.state, obs_publish)
        sim.do_simulation(ai, di, mpc.cx, mpc.cy, mpc.cyaw, mpc.target_ind)


    elapsed_time = time.time() - start
    print(f"calc time:{elapsed_time:.6f} [sec]")


if __name__ == '__main__':
    main()
