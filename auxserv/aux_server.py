#!/usr/bin/env python


### configuration section

MECOM_CONFIG = [
    ('AMP1', "/dev/ttyUSB1", 26.5),
    ('AMP2', "/dev/ttyUSB2", 24.5),
    ('AMP3', "/dev/ttyUSB3", 24.5)
    ]

DT=0.25
SERVER="localhost"
PORT=23000

#### code section

from mecom import MeCom
import time
import datetime
import socket
import socketserver
import threading

class BMX_Temp:

    def __init__ (self,bmxid, serialport, target_temp):
        mc=MeCom(serialport)
        self.mc=mc
        self.serialport=serialport
        self.ID=bmxid
        self.temp=[-1,-1]

        # which device are we talking to?
        self.address = mc.identify()
        self.status = mc.status()

        # Display model number
        print("Controller number:", bmxid, "connected to device: {}, self.status: {}".format(self.address, self.status))
        model = mc.get_parameter(parameter_name="Device Type", address=self.address)
        print("Controller number:", bmxid, "Device Type: {}".format(model))

        # Set voltage limit to 8.0v for CP60233 TEC element
        success = mc.set_parameter(value=1.0, parameter_name="Voltage Limitation")
        print("Controller number:", bmxid, "Max V set?: {}".format(success))

        # Set Curremt limit to 6.0A for CP60233 TEC element
        success = mc.set_parameter(value=.5, parameter_name="Current Limitation")
        print("Controller number:", bmxid, "Max I set?: {}".format(success))

        # Set sink temperature to fixed or external
        success = mc.set_parameter(value=1, parameter_name="Sink Temperature Selection")    #0=External, 1=fixed value
        print("Controller number:", bmxid, "Sink temperature mode set?: {}".format(success))

        # Set object sensor type
        success = mc.set_parameter(value=1, parameter_name="Sensor Type Selection")    #0=NTC, 1=PT100, 2=PT1k
        print("Controller number:", bmxid, "Sensor Type Selection Set?: {}".format(success))

        # Set voltage and current (fixed power mode)
        #myV = 6.0
        #myI = 0.5
        #success = mc.set_parameter(value=myV, parameter_name="Set Voltage")
        #print("Voltage set to {}?:{}".format(myV, success))
        #success = mc.set_parameter(value=myI, parameter_name="Set Current")
        #print("Current set to {}?:{}".format(myI, success))

        # Write data to Flash
        success = mc.set_parameter(value=0, parameter_name="Save Data to Flash")    #0=enable, 1=disable
        print("Controller number:", bmxid, "Saved to flash?:{}".format(success))

        success = mc.set_parameter(value=0, parameter_name="Input Selection", parameter_instance = 1)       #0=static, 1=live, 2=temperature controller
        print("Controller number:", bmxid, "Mode set: {}".format(success))
        
        success = mc.set_parameter(value=2, parameter_name="Input Selection", parameter_instance = 2)       #0=static, 1=live, 2=temperature controller
        print("Controller number:", bmxid, "Mode 2 set: {}".format(success))

        #set target temperature
        for pi in [1,2]:
            success = mc.set_parameter(value=target_temp, parameter_id=3000, parameter_instance = pi)
            print("Controller number:", bmxid, "Target temperature set: {}".format(success))

        # Operating mode
        success = mc.set_parameter(value=0, parameter_name="Status")    #0=Static OFF, 1=Static ON, 2=Live OFF/ON
        print("Controller number:", bmxid, "Operating mode set: {}".format(success))

        # Operating mode
        success = mc.set_parameter(value=1, parameter_name="Status", parameter_instance = 2)    #0=Static OFF, 1=Static ON, 2=Live OFF/ON
        print("Controller number:", bmxid, "Operating mode set 2: {}".format(success))

    def control(self):
        mc=self.mc

        for pi in [1,2]:
            # get object temperature
            self.temp[pi-1] = mc.get_parameter(parameter_name="Object Temperature", address=self.address, parameter_instance = pi)
            print("Controller number:", self.ID, "Object Temperature: {}C".format(self.temp[pi-1]))

            # get sink temperature
            temp = mc.get_parameter(parameter_name="Sink Temperature", address=self.address, parameter_instance = pi)
            print("Controller number:", self.ID, "Sink Temperature: {}C".format(temp))


            temp = mc.get_parameter(parameter_id=3000, address=self.address, parameter_instance = pi)
            print("Controller number:", self.ID, "Target temp 2: {}C".format(temp))


            # is the control loop active, and if it is, is it stable?
            stable_id = mc.get_parameter(parameter_name="Temperature is Stable", address=self.address, parameter_instance = pi)
            if stable_id == 0:
                stable = "temperature regulation is not active"
            elif stable_id == 1:
                stable = "is not stable"
            elif stable_id == 2:
                stable = "is stable"
            else:
                stable = "state is unknown"
            print("Controller number:", self.ID, "query for loop stability, loop {}".format(stable))
 
        self.status = mc.status()
        print("Controller number:", self.ID, "Status: {}".format(self.status))
       

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):

    def handle(self):
        while True:
            try:
                command = str(self.request.recv(1024), 'ascii')
                cur_thread = threading.current_thread()
                if command=="LIST":
                    response="  ".join([m.ID for m in mcs])
                elif command=="DATA":
                    response=""
                    for mc in mcs:
                        response+="{} {}".format(mc.temp[0],mc.temp[1])
                elif command=="BYE":
                    break
                else:
                    print ("UNKNOWN COMMAND RECEIVED!")
                    response=""
                self.request.sendall(bytes(response,'ascii'))
            except:
                break
        print ("Exiting...")

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


if __name__ == "__main__":

    mcs = [BMX_Temp(*params) for params in MECOM_CONFIG]
    time.sleep(1) 
    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    ip, port = server.server_address

    # Start a thread with the server -- that thread will then start one
    # more thread for each request
    server_thread = threading.Thread(target=server.serve_forever)
    # Exit the server thread when the main thread terminates
    server_thread.daemon = True
    server_thread.start()
    print("Server loop running in thread:", server_thread.name)

    try:
        while True:
            for mc in mcs:
                mc.control()
                time.sleep(DT)

    except KeyboardInterrupt:
        print('Shutting down server...')
        try:
            server.shutdown()
            server.server_close()
            print ("Shutdown complete")
        except SystemExit:
            os._exit(0)
