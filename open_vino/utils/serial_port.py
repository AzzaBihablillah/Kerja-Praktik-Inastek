import serial
import serial.tools.list_ports


class Micro:
    def __init__(self, baudrate=9600, timeout=20):
        self.baudrate = baudrate
        self.timeout = timeout
        # try:
        #     self.port = self.find_serial_port()
        #     self.micro = serial.Serial(
        #         port=self.port, baudrate=self.baudrate, timeout=self.timeout
        #     )
        # except serial.SerialException as e:
        #     print(f'Failure during micro device initialization. Reason: {e}')
        #     # print("No micro devices connected!")

    def find_serial_port(self):
        # self.ports = serial.tools.list_ports.comports()
        # port = [
        #     port.device for port in self.ports if "USB" in port.description.split()
        # ][0]

        # return port
        pass

    def send_command(self, command: str):
        if command == "benda_masuk":
            print(f'command: A')
            # self.micro.write(b"A\n")
            return

        elif command == "tolak_benda":
            print(f'command: B')
            # self.micro.write(b"B\n")
            return

        elif command == "bottle":
            print(f'command: C')
            # self.micro.write(b"C\n")
            return
        
        else:
            print(f'Invalid command')
    
    def debug_send_command(self,command):
        print(f'Command sent: {command}')
        return

    def read_serial_message(self):
        # return self.micro.readline().decode().strip()
        pass
    
    def debug_read_serial_message(self,debug_message=''):
        # return self.micro.readline().decode().strip()
        if debug_message == '':
            print(f'Placeholder read serial message. Please put something in debug_message argument')
            return debug_message
        else:
            print(f'Debug: sent \"{debug_message}\"')
            return debug_message


if __name__ == "__main__":
    # micro = Micro(baudrate=115200, timeout=1)
    # micro.send_command("benda_masuk")
    # print(micro.micro)
    pass
