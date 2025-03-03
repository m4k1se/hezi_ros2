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
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

# from utils.angle import angle_mod


NX = 4  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]
T = 10  # horizon length

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

def smooth_yaw_iter(previous_yaw, new_yaw):
    """
    Smooth the yaw angle based on the previous yaw to ensure continuity.

    :param previous_yaw: (float) Previous yaw angle in radians
    :param new_yaw: (float) New yaw angle in radians (not yet normalized)
    :return: (float) Smoothed and normalized yaw angle in radians within (-pi, pi]
    """
    dyaw = new_yaw - previous_yaw

    # 调整dyaw，使其在[-pi, pi]范围内
    dyaw = (dyaw + np.pi) % (2.0 * np.pi) - np.pi

    # 平滑后的yaw
    smoothed_yaw = previous_yaw + dyaw
    # print(f"previous_yaw:{previous_yaw}, new_yaw:{new_yaw}, smoothed_yaw:{smoothed_yaw}")
    return smoothed_yaw

class Simulator:
    
    def __init__(self, initial_state):
        self.state = initial_state
        self.x = [self.state.x]
        self.y = [self.state.y]
        self.yaw = [self.state.yaw]
        self.v = [self.state.v]
    
    def update_state(self, a, delta):

        # input check
        if delta >= MAX_STEER:
            delta = MAX_STEER
        elif delta <= -MAX_STEER:
            delta = -MAX_STEER
        
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
    
    def do_simulation(self, ai, di, cx, cy, cyaw, target_ind):
        
        if self.state.yaw - cyaw[0] >= math.pi:
            self.state.yaw -= math.pi * 2.0
        elif self.state.yaw - cyaw[0] <= -math.pi:
            self.state.yaw += math.pi * 2.0
        print("state.yaw:", self.state.yaw)
        cyaw = self.smooth_yaw(cyaw)
        self.update_state(ai, di)
        
        self.x.append(self.state.x)
        self.y.append(self.state.y)
        self.yaw.append(self.state.yaw)
        self.v.append(self.state.v)

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(cx, cy, "-r", label="course")
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
            
class MPCfollower:
    def __init__(self, path):
        self.cx, self.cy, self.cyaw, self.ck = read_csv(path)
        self.sp = self.calc_speed_profile(TARGET_SPEED)
        self.episode = 1
        self.oa = None
        self.odelta = None
        self.ai = 0
        self.di = 0
        
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
    
    def calc_ref_trajectory(self, state, dl, pind):
        xref = np.zeros((NX, T + 1))
        dref = np.zeros((1, T + 1))
        ncourse = len(self.cx)

        ind, _ = self.calc_nearest_index(state, pind)
        # print("最近点的索引：", ind, pind)
        if pind >= ind:
            ind = pind

        xref[0, 0] = self.cx[ind]
        xref[1, 0] = self.cy[ind]
        xref[2, 0] = self.sp[ind]
        xref[3, 0] = self.cyaw[ind]
        dref[0, 0] = 0.0  # steer operational point should be 0

        travel = 0.0

        for i in range(T + 1):
            travel += abs(state.v) * DT  # 累计形式的距离
            dind = int(round(travel / dl))  # dl是路径点的间隔，travel/dl是当前车辆已经行驶的路径点数

            # if (ind + dind) < ncourse:  #n course是路径点的总数，判断是否超出路径点的总数
                # xref[0, i] = cx[ind + dind]
                # xref[1, i] = cy[ind + dind]
                # xref[2, i] = sp[ind + dind]
                # xref[3, i] = cyaw[ind + dind]
                # dref[0, i] = 0.0
            if (ind + i) < ncourse:
                xref[0, i] = self.cx[ind + i]
                xref[1, i] = self.cy[ind + i]
                xref[2, i] = self.sp[ind + i]
                xref[3, i] = self.cyaw[ind + i]
                dref[0, i] = 0.0
            else:
                xref[0, i] = self.cx[ncourse - 1]
                xref[1, i] = self.cy[ncourse - 1]
                xref[2, i] = self.sp[ncourse - 1]
                xref[3, i] = self.cyaw[ncourse - 1]
                dref[0, i] = 0.0

        return xref, ind, dref
    
    def cal_acc_and_delta(self, state):
        if self.episode == 1:
            self.target_ind, _ = self.calc_nearest_index(state, 0)
        xref, self.target_ind, dref = self.calc_ref_trajectory(state, 1.0, self.target_ind)
        x0 = [state.x, state.y, state.v, state.yaw]
        self.oa, self.odelta, ox, oy, oyaw, ov = self.iterative_linear_mpc_control(xref, x0, dref, self.oa, self.odelta)
        if self.odelta is not None:
            new_di, self.ai = self.odelta[0], self.oa[0]
            self.di = smooth_yaw_iter(self.di, new_di)
            print(f"di:{self.di}, ai:{self.ai}")
            # di 大于max，就取max，小于min，就取min
            if self.di >= MAX_STEER:
                self.di = MAX_STEER
            elif self.di <= -MAX_STEER:
                self.di = -MAX_STEER
        self.episode += 1
        return self.ai, self.di
        
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
    start_x = mpc.cx[0]
    start_y = mpc.cy[0]
    start_yaw = mpc.cyaw[0]
    initial_state = State(x=start_x, y=start_y, yaw=start_yaw, v=0.0)
    sim = Simulator(initial_state)
    
    for i in range(500):
        ai, di = mpc.cal_acc_and_delta(sim.state)
        sim.update_state(ai, di)
        sim.do_simulation(ai, di, mpc.cx, mpc.cy, mpc.cyaw, mpc.target_ind)


    elapsed_time = time.time() - start
    print(f"calc time:{elapsed_time:.6f} [sec]")


if __name__ == '__main__':
    main()
