import socket

# UDP_IP = "0.0.0.0"
UDP_IP = "127.0.0.1"
UDP_PORT = 5002
sock = socket.socket(socket.AF_INET,  # Internet
                     socket.SOCK_DGRAM)  # UDP
sock.bind((UDP_IP, UDP_PORT))

print(f'Listening on {UDP_IP} at port {UDP_PORT}') 
while True:
    data, addr = sock.recvfrom(1024)  # buffer size is 1024 bytes
    print(f'received message: {data}')
