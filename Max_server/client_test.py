import socket

UDP_IP = "212.11.40.145"
# UDP_IP = "63.33.36.17"
UDP_PORT = 5001
MESSAGE = "Yo"

print(f'Sending {MESSAGE} at {UDP_IP} on port {UDP_PORT}')
sock = socket.socket(socket.AF_INET,
                     socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.sendto(str.encode(MESSAGE), (UDP_IP, UDP_PORT))
