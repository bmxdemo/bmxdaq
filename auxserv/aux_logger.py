#!/usr/bin/env python
import socket,sys,time

SERVER="localhost"
PORT=23000

def client(ip, port, filename):
    f=open(filename,'w')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    try:
        sock.sendall(bytes("LIST", 'ascii'))
        response = str(sock.recv(1024), 'ascii')
        print ("SENSORS: ",response)

        while True:
            sock.sendall(bytes("DATA", 'ascii'))
            response = str(sock.recv(1024), 'ascii')
            print("Received: {}".format(response))
            now = datetime.datetime.now()
            f.write("%i %i %i %i %i %i "%(now.year, now.month, now.day, now.hour, now.minute, now.second))
            f.write(response)
            f.write("\n")
            f.flush()
            time.sleep(1)
    except KeyboardInterrupt:
        sock.sendall(bytes("BYE", 'ascii'))
        sock.close()


if __name__ == "__main__":
    # Port 0 means to select an arbitrary unused port
    HOST, PORT = "localhost", 23000

    client("localhost", PORT, 'output.dat')

