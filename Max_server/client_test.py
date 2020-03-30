import socket

UDP_IP = "212.11.40.145"
UDP_PORT = 5001
MESSAGE = "Yo"

print(f'Sending {MESSAGE} at {UDP_IP} on port {UDP_PORT}')
sock = socket.socket(socket.AF_INET,  # Internet
                     socket.SOCK_DGRAM)  # UDP
sock.sendto(str.encode(MESSAGE), (UDP_IP, UDP_PORT))
