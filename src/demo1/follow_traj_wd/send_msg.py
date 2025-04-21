import can
import time
import random

def create_bus(channel):
    return can.interface.Bus(channel=channel, bustype='socketcan')

def send_message(bus, arbitration_id, data):
    msg = can.Message(arbitration_id=arbitration_id, data=data, is_extended_id=False)
    try:
        bus.send(msg)
        print(f"Sent {hex(arbitration_id)}: {data}")
    except can.CanError:
        print(f"Message NOT sent on {hex(arbitration_id)}")

def main():
    bus_ins = create_bus('can0')  # 模拟INS设备
    bus_vcu = create_bus('can1')  # 模拟VCU设备

    while True:
        # --- vcan0 INS 报文 ---
        # 纬经度 (0x504) 经转换为约118.8171577, 31.8925019
        lat = int((31.8925895 + 180) / 0.0000001)
        lon = int((118.8171084 + 180) / 0.0000001)
        data_504 = [
            (lat >> 24) & 0xFF, (lat >> 16) & 0xFF, (lat >> 8) & 0xFF, lat & 0xFF,
            (lon >> 24) & 0xFF, (lon >> 16) & 0xFF, (lon >> 8) & 0xFF, lon & 0xFF
        ]
        send_message(bus_ins, 0x504, data_504)

        # 北向速度、东向速度、地向速度（0x505）
        def encode_speed(val):
            spd = int((val / 3.6 + 100) / 0.0030517)
            return [(spd >> 8) & 0xFF, spd & 0xFF]
        data_505 = encode_speed(10) + encode_speed(5) + encode_speed(3)
        send_message(bus_ins, 0x505, data_505)

        # 航向角（0x502）
        heading = int((-92.27118000000002 + 360) / 0.010986)
        data_502 = [0, 0, 0, 0, (heading >> 8) & 0xFF, heading & 0xFF, 0, 0]
        send_message(bus_ins, 0x502, data_502)

        # 加速度（0x500）
        acc = int(((1 / 9.8 + 4) / 0.0001220703125))
        data_500 = [(acc >> 8) & 0xFF, acc & 0xFF, 0, 0, 0, 0, 0, 0]
        send_message(bus_ins, 0x500, data_500)

        # --- vcan1 VCU 报文 ---
        # 自动驾驶允许（0x15C）
        data_15C = [0x00, 0x00, 0x01, 0, 0, 0, 0, 0]
        send_message(bus_vcu, 0x15C, data_15C)

        # EPS模式（0x124），假设模式为2
        mode = 2 << 4
        data_124 = [0, 0, 0, 0, 0, 0, mode, 0]
        send_message(bus_vcu, 0x124, data_124)

        time.sleep(1)

if __name__ == "__main__":
    main()
