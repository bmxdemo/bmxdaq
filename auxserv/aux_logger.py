#!/usr/bin/env python
import socket,sys,time, datetime

SERVER="localhost"
PORT=23000

def client(ip, port, filename):
    f=open(filename,'w')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    try:
        sock.sendall(bytes("LIST",'ascii'))
        response = str(sock.recv(1024), 'ascii')
        print ("SENSORS: ",response)

        while True:
            sock.sendall(bytes("DATA",'ascii'))
            response = str(sock.recv(1024), 'ascii')
            print("Received: {}".format(response))
            now = datetime.datetime.now()
            f.write("%i %i %i %i %i %i "%(now.year, now.month, now.day, now.hour, now.minute, now.second))
            f.write(response)
            f.write("\n")
            f.flush()
            time.sleep(1)
    except KeyboardInterrupt:
        print("SENDING BYE")
        sock.sendall(bytes("BYE",'ascii'))
        print ("Closing socket.")
        sock.close()


if __name__ == "__main__":
    client(SERVER, PORT, 'output.dat')
    print ("Exiting client.")
