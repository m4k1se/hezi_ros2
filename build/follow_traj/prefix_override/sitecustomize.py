import sys
if sys.prefix == '/home/renth/anaconda3/envs/demo1':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/renth/mpc_ws/install/follow_traj'
