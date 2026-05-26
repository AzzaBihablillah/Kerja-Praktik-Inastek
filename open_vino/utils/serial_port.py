import serial
import serial.tools.list_ports
from enum import Enum
from typing import Union, List


# class Micro:
#     def __init__(self, baudrate=9600, timeout=20):
#         self.baudrate = baudrate
#         self.timeout = timeout
#         try:
#             self.port = self.find_serial_port()
#             self.micro = serial.Serial(
#                 port=self.port, baudrate=self.baudrate, timeout=self.timeout
#             )
#         except serial.SerialException as e:
#             print(f'Failure during micro device initialization. Reason: {e}')
#             # print("No micro devices connected!")

#     def find_serial_port(self):
#         self.ports = serial.tools.list_ports.comports()
#         port = [
#             port.device for port in self.ports if "USB" in port.description.split()
#         ][0]

#         return port
#         pass

#     def send_command(self, command: str):
#         if command == "benda_masuk":
#             print(f'command: A')
#             # self.micro.write(b"A\n")
#             return

#         elif command == "tolak_benda":
#             print(f'command: B')
#             # self.micro.write(b"B\n")
#             return

#         elif command == "bottle":
#             print(f'command: C')
#             # self.micro.write(b"C\n")
#             return
        
#         else:
#             print(f'Invalid command')
    
#     # def debug_send_command(self,command):
#     #     print(f'Command sent: {command}')
#     #     return

#     def read_serial_message(self):
#         # return self.micro.readline().decode().strip()
#         pass
    
#     # def debug_read_serial_message(self,debug_message=''):
#     #     # return self.micro.readline().decode().strip()
#     #     if debug_message == '':
#     #         print(f'Placeholder read serial message. Please put something in debug_message argument')
#     #         return debug_message
#     #     else:
#     #         print(f'Debug: sent \"{debug_message}\"')
#     #         return debug_message

import serial
import serial.tools.list_ports
import logging
from enum import Enum

logger = logging.getLogger(__name__)

SPECIAL_PORTS = {"loop://", "spy://", "socket://"}

class Command(Enum):
    BENDA_MASUK = "A"
    TOLAK_BENDA = "B"
    BOTTLE      = "C"


class Micro:
    def __init__(self, baudrate: int = 9600, timeout: float = 20.0):
        self.baudrate = baudrate
        self.timeout  = timeout
        self.micro    = None
        # self.connected = False

        # port = self._find_serial_port()
        # self.micro = serial.Serial(
        #     port=port, baudrate=self.baudrate, timeout=self.timeout
        # )
        # logger.info(f"Connected to serial device on {port}")

    def list_serial_port(self) -> List[str]:
        """
        Scan available ports and return the first USB serial device found.

        :raises RuntimeError: If required is True and no USB serial device is connected.
        """
        ports = serial.tools.list_ports.comports()
        usb_ports = [
            p.device for p in ports if "USB" in p.description.upper()
        ]
        return usb_ports
    
    @property
    def current_port(self) -> str | None:
        if self.micro is not None and self.micro.is_open:
            return self.micro.port
        return None
    
    def open(self, 
             port: str | int = 0) -> None:
        if self.micro is not None:
            logger.info("A port is already open, closing before reopening...")
            self.close()

        available_ports = self.list_serial_port()

        if isinstance(port, str):
            if port not in available_ports and port not in SPECIAL_PORTS:
                logger.warning(f"Port {port} is not connected to this device, failure to open port")
                return None
            self.micro = serial.Serial(
                port=port, baudrate=self.baudrate, timeout=self.timeout
            )
            logger.info(f"Port {port} has been opened.")
        elif isinstance(port, int):
            if not available_ports:
                logger.warning("No serial ports detected, failure to open port.")
                return None
            elif port >= len(available_ports):
                logger.warning("Index out of bounds, failure to open port.")
                return None
            else:
                self.micro = serial.Serial(
                port=available_ports[port], baudrate=self.baudrate, timeout=self.timeout
            )
                logger.info(f"Port {available_ports[port]} has been opened.")
        else:
            logger.warning(f"Accepted parameter: str or int. You entered: {type(port)}")

    def reconnect(self) -> bool:
        last_port = self.current_port
        if last_port is None:
            logger.warning("No previous port to reconnect to.")
            return False
        self.close()
        self.open(last_port)
        return True

    def send_command(self, command: Command) -> None:
        """
        Send a command to the microcontroller.

        :param command: A Command enum value.
        :raises TypeError: If command is not a Command enum.
        :raises serial.SerialException: If write fails.
        """
        if self.micro is None:
            logger.warning(f"No port is opened! Use open() method first to use this function.")
            return None
        if not isinstance(command, Command):
            raise TypeError(
                f"Expected a Command enum, got {type(command).__name__}. "
                f"Valid commands: {[c for c in Command]}"
            )
        payload = f"{command.value}\n".encode()
        self.micro.write(payload)
        logger.debug(f"Sent command: {command.name} ({command.value})")

    def read_message(self) -> str | None:
        """
        Read one line from the microcontroller.

        :return: Decoded and stripped message string or None if no message is received.
        """
        if self.micro is None:
            logger.warning(f"No port is opened! Use open() method first to use this function.")
            return None
        
        raw = self.micro.readline()
        if not raw:
            logger.debug(f"No message received within {self.timeout}s timeout.")
            return None
            # raise serial.SerialTimeoutException(
            #     f"No message received within {self.timeout}s timeout."
            # )
        return raw.decode().strip()

    def close(self) -> None:
        """Release the serial port."""
        if self.micro and self.micro.is_open:
            self.micro.close()
            self.micro = None
            logger.info("Serial port closed.")

    def check_status(self) -> None:
        """
        Printing the status of Micro, listing connected ports and which port is opened by this class.
        """
        print(f"=====================[Status]=====================")
        available_ports = self.list_serial_port()
        print(f"List of connected ports: {available_ports if available_ports else "No connected ports"}")
        print(f"Connected port: {self.current_port if self.current_port else "Disconnected"}")
        print(f"==================================================")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    



if __name__ == "__main__":
    micro = Micro()
    available_ports = micro.list_serial_port()
    micro.open(available_ports[0])
    micro.check_status()
    micro.close()
    micro.check_status()

    
    # pass
