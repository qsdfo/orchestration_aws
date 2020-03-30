import socket
TCP_IP = '63.33.36.17'
TCP_PORT = 5003
BUFFER_SIZE = 1024
MESSAGE = str.encode("Yo TCP")

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))
s.send(MESSAGE)
data = s.recv(BUFFER_SIZE)
s.close()
print(f'received data: {data}')
