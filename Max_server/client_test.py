import socket

UDP_IP = "63.33.36.17"
UDP_PORT = 5002
MESSAGE = "Yo"

print(f'Sending {MESSAGE} at {UDP_IP} on port {UDP_PORT}')
sock = socket.socket(socket.AF_INET,  # Internet
                     socket.SOCK_DGRAM)  # UDP
sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
