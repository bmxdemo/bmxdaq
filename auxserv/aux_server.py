#!/usr/bin/env python


####### CONFIGURATION SECTION #######
#####################################
tempconON = 0   	   #Turn active temperature control on or off (1=ON, 0=OFF)
DT=1.0                    #Delay between temperature monitoring prints

Kp = 10.0     #PID loop P coefficient			
Ti = 100.0    #PID loop I coefficient               Changed from 300 to 100  9/4/18
Td = 1.0      #PID loop D coefficient

HOST="localhost"
PORT=23000

MECOM_CONFIG = [
    ('AMP1', "/dev/ttyUSB0", 18.0),     	#Sets controller, USB port, and temperature set point
    ('AMP2', "/dev/ttyUSB1", 18.0),
    ('AMP3', "/dev/ttyUSB2", 18.0),
    #('TERMS', "/dev/ttyUSB3", 26.5)	#What is this? [jh]
]
######################################
#### END OF CONFIGURATION SECTION ####



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

        if tempconON == 1:
        	mode_set=2
        	status_set=1
        elif tempconON == 0:
        	mode_set=0
        	status_set=0

        # Display model number
        print("Controller", bmxid, "connected to device: {}, self.status: {}".format(self.address, self.status))
        model = mc.get_parameter(parameter_name="Device Type", address=self.address)
        print("Controller", bmxid, "Device Type: {}".format(model))

        # Set voltage limit for CP60233 TEC element
        success = mc.set_parameter(value=1.0, parameter_name="Voltage Limitation")
        print("Controller", bmxid, "Max V set?: {}".format(success))

        # Set Curremt limit for CP60233 TEC element
        success = mc.set_parameter(value=1.0, parameter_name="Current Limitation")
        print("Controller", bmxid, "Max I set?: {}".format(success))

        # Set sink temperature to fixed or external
        success = mc.set_parameter(value=1, parameter_name="Sink Temperature Selection")    #0=External, 1=fixed value
        print("Controller", bmxid, "Sink temperature mode set?: {}".format(success))

        # Set object sensor type
        success = mc.set_parameter(value=1, parameter_name="Sensor Type Selection")    #0=NTC, 1=PT100, 2=PT1k
        print("Controller", bmxid, "Sensor Type Selection Set?: {}".format(success))

        # Get current PID Values (10, 300, 0 on 9/4/18)
        #Kp_out = mc.get_parameter(parameter_name="Kp", address=self.address, parameter_instance = 2)
        #print("Controller", self.ID, "Current Kp for channel 2:", Kp_out)
        #Ti_out = mc.get_parameter(parameter_name="Ti", address=self.address, parameter_instance = 2)
        #print("Controller", self.ID, "Current Kp for channel 2:", Ti_out)
        #Td_out = mc.get_parameter(parameter_name="Td", address=self.address, parameter_instance = 2)
        #print("Controller", self.ID, "Current Kp for channel 2:", Td_out)

        # Set PID Values
        success = mc.set_parameter(value=Kp, parameter_name="Kp", parameter_instance = 1)
        print("Controller", bmxid, "P value for channel 1 set?: {}".format(success))
        success = mc.set_parameter(value=Ti, parameter_name="Ti", parameter_instance = 1)
        print("Controller", bmxid, "I value for channel 1 set?: {}".format(success))
        success = mc.set_parameter(value=Td, parameter_name="Td", parameter_instance = 1)
        print("Controller", bmxid, "D value for channel 1 set?: {}".format(success))
        success = mc.set_parameter(value=Kp, parameter_name="Kp", parameter_instance = 2)
        print("Controller", bmxid, "P value for channel 2 set?: {}".format(success))
        success = mc.set_parameter(value=Ti, parameter_name="Ti", parameter_instance = 2)
        print("Controller", bmxid, "I value for channel 2 set?: {}".format(success))
        success = mc.set_parameter(value=Td, parameter_name="Td", parameter_instance = 2)
        print("Controller", bmxid, "D value for channel 2 set?: {}".format(success))

        # Write data to Flash
        success = mc.set_parameter(value=0, parameter_name="Save Data to Flash")    #0=enable, 1=disable
        print("Controller", bmxid, "Saved to flash?:{}".format(success))

        if bmxid == "TERMS":

            success = mc.set_parameter(value=mode_set, parameter_name="Input Selection", parameter_instance = 1)       #0=static, 1=live, 2=temperature controller
            print("Controller", bmxid, "Mode for channel 1 set: {}".format(success))

            success = mc.set_parameter(value=mode_set,  parameter_name="Input Selection", parameter_instance = 2)       #0=static, 1=live, 2=temperature controller
            print("Controller", bmxid, "Mode for channel 2 set: {}".format(success))

        else:
            success = mc.set_parameter(value=mode_set, parameter_name="Input Selection", parameter_instance = 1)       #0=static, 1=live, 2=temperature controller
            print("Controller", bmxid, "Mode for channel 1 set: {}".format(success))

            success = mc.set_parameter(value=mode_set, parameter_name="Input Selection", parameter_instance = 2)       #0=static, 1=live, 2=temperature controller
            print("Controller", bmxid, "Mode for channel 2 set: {}".format(success))

        #set target temp
        for ch in [1,2]:
            success = mc.set_parameter(value=target_temp, parameter_id=3000, parameter_instance = ch)
            print("Controller", bmxid, "Target temperature for channels 1 and 2 set: {}".format(success))


        if bmxid == "TERMS":
        # Operating mode
            success = mc.set_parameter(value=status_set, parameter_name="Status")    #0=Static OFF, 1=Static ON, 2=Live OFF/ON
            print("Controller", bmxid, "Operating mode for channel 1 set: {}".format(success))

            # Operating mode
            success = mc.set_parameter(value=status_set, parameter_name="Status", parameter_instance = 2)    #0=Static OFF, 1=Static ON, 2=Live OFF/ON
            print("Controller", bmxid, "Operating mode for channel 2 set: {}".format(success))
        else:
            # Operating mode
            success = mc.set_parameter(value=status_set, parameter_name="Status")    #0=Static OFF, 1=Static ON, 2=Live OFF/ON
            print("Controller", bmxid, "Operating mode for channel 1 set: {}".format(success))

            # Operating mode
            success = mc.set_parameter(value=status_set, parameter_name="Status", parameter_instance = 2)    #0=Static OFF, 1=Static ON, 2=Live OFF/ON
            print("Controller", bmxid, "Operating mode for channel 2 set 2: {}".format(success))


    def control(self):		#This is the main control loop
        mc=self.mc
        try:

            for ch in [1,2]:		#For both channels on each controller board
                # get object temperature
                self.temp[ch-1] = mc.get_parameter(parameter_name="Object Temperature", address=self.address, parameter_instance = ch)
                print("Ch.", ch, self.ID, "Object Temperature: {}C".format(self.temp[ch-1]))

                # get sink temperature
                #temp = mc.get_parameter(parameter_name="Sink Temperature", address=self.address, parameter_instance = ch)
                #print("Controller", self.ID, "Sink Temperature: {}C".format(temp))

                temp = mc.get_parameter(parameter_name="Target Object Temp (Set)", address=self.address, parameter_instance = ch)
                print("Ch.", ch, self.ID, "Target temp: {}C".format(temp))

                #if self.ID == "TERMS":
                #    #reset target temp
                #    temp =  mc.get_parameter(parameter_id = 3000, address=self.address, parameter_instance = ch)
                #    if temp == 20.0:
                #        temp = mc.set_parameter(parameter_id = 3000, address=self.address, parameter_instance = ch, value = 26.0)
                #    else:
                #        temp = mc.set_parameter(parameter_id = 3000, address=self.address, parameter_instance = ch, value = 20.0)

                # is the control loop active, and if it is, is it stable?
                stable_id = mc.get_parameter(parameter_name="Temperature is Stable", address=self.address, parameter_instance = ch)
                if stable_id == 0:
                    stable = "is not active"
                elif stable_id == 1:
                    stable = "is not stable"
                elif stable_id == 2:
                    stable = "is stable"
                else:
                    stable = "state is unknown"
                print("Ch.", ch, self.ID, "Control loop {}".format(stable))

                self.status = mc.status()
                print("Ch.", ch, self.ID, "Status: {}".format(self.status))

        except:
            print ("Caught exception!")
            pass
        #if self.ID == "TERMS":
            # time.sleep(600)



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
                        response+="{} {} ".format(mc.temp[0],mc.temp[1])
                elif command=="BYE":
                    print ("Received BYE")
                    break
                else:
                    print ("UNKNOWN COMMAND RECEIVED! >"+command+"<")
                    response=""
                    break
                self.request.sendall(bytes(response,'ascii'))
            except:
                break
        print ("Exiting server on thread %s..."%cur_thread.name)

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


if __name__ == "__main__":

    mcs = [BMX_Temp(*params) for params in MECOM_CONFIG]
    print("Starting monitoring and control...\n")
    time.sleep(2)
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
        print("\n\n")		#Leave some space between the prints from the set of three controllers.
        while True:
            for mc in mcs:      #for each controller in mcs
                mc.control()
                print("\n")

            time.sleep(DT)

    except KeyboardInterrupt:
        print('Shutting down server...')
        try:
            server.shutdown()
            server.server_close()
            print ("Shutdown complete")
        except SystemExit:
            os._exit(0)
