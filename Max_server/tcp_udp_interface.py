import argparse
import pickle
import socket

import numpy as np


class TCP_UDP_interface(object):
    def __init__(self):
        self.client_tcp = None
        self.server_udp = None  # pour l'instant


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port_tcp', type=int, default=5001)
    parser.add_argument('--port_udp', type=int, default=5002)
    args = parser.parse_args()

    HOST, PORT = "127.0.0.1", args.port_tcp
    np_array = np.random.rand(3, 2)
    data = dict(
        np_array=np_array,
        function='orchestrate'
    )
    message = pickle.dumps(data)
    # Create a socket (SOCK_STREAM means a TCP socket)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Connect to server and send data
        sock.connect((HOST, PORT))
        sock.sendall(message)
        # Receive data from the server and shut down
        received = str(sock.recv(1024), "utf-8")

    print("Sent:     {}".format(data))
    print("Received: {}".format(received))
