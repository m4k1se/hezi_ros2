import can
import time

def main():
    bus_ins = can.interface.Bus(channel='vcan0', interface='socketcan')
    bus_vcu = can.interface.Bus(channel='vcan1', interface='socketcan')
    while True:
        msg_loc = can.Message(arbitration_id=0x504, 
                            data=[0x00, 0x00, 0x00, 0x01, 
                                    0x00, 0x00, 0x00, 0x01],
                            is_extended_id=False)
        bus_ins.send(msg_loc)
        print("Sent location message on vcan0")
        
        # msg_yaw = can.Message(arbitration_id=0x502, 
        #                         data=[0x00, 0x01, 0x00, 0x02, 
        #                             0x00, 0x03, 0x00, 0x00],
        #                         is_extended_id=False)
        # bus_ins.send(msg_yaw)
        # print("Sent yaw message on vcan0")
        
        # msg_acc = can.Message(arbitration_id=0x500, 
        #                         data=[0x00, 0x01, 0x00, 0x02, 
        #                             0x00, 0x03, 0x00, 0x00],
        #                         is_extended_id=False)
        # bus_ins.send(msg_acc)
        # print("Sent acceleration message on vcan0")
        
        msg_vcu = can.Message(arbitration_id=0x15C, 
                            data=[0x00, 0x00, 0x01, 0x00, 
                                    0x00, 0x00, 0x00, 0x00],
                            is_extended_id=False)
        bus_vcu.send(msg_vcu)
        print("Sent VCU message on vcan1")
        
        time.sleep(0.05)

if __name__ == "__main__":
    main()