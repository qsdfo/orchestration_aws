# import argparse
import pickle
import socket

# Helper function to parse attribute
import struct

from pythonosc import dispatcher, osc_server
from pythonosc.udp_client import SimpleUDPClient


def osc_attr(obj, attribute):
    def closure(*args):
        args = args[1:]
        if len(args) == 0:
            return getattr(obj, attribute)
        else:
            return setattr(obj, attribute, *args)

    return closure


class OSCServer(object):
    """
    Key class for OSCServers linking Python and Max / MSP

    Example :
    >>> server = OSCServer(1234, 1235) # Creating server
    >>> server.run() # Running server
    """
    # attributes automatically bounded to OSC ports
    osc_attributes = []

    # Initialization method
    def __init__(self, in_port, out_port, ip, *args):
        super(OSCServer, self).__init__()
        # OSC library objects
        self.dispatcher = dispatcher.Dispatcher()
        self.client = SimpleUDPClient(ip, out_port)
        # Bindings for server
        self.init_bindings(self.osc_attributes)
        self.server = osc_server.BlockingOSCUDPServer((ip, in_port), self.dispatcher)
        self.server.allow_reuse_address = True
        # Server properties
        self.debug = False
        self.in_port = in_port
        self.out_port = out_port
        self.ip = ip

    def init_bindings(self, osc_attributes=[]):
        """Here we define every OSC callbacks"""
        self.dispatcher.map("/ping", self.ping)
        self.dispatcher.map("/stop", self.stopServer)
        for attribute in osc_attributes:
            print(attribute)
            self.dispatcher.map("/%s" % attribute, osc_attr(self, attribute))

    def stopServer(self, *args):
        """stops the server"""
        self.client.send_message("/terminated", "bang")
        self.server.shutdown()
        self.server.socket.close()

    def run(self):
        """runs the SoMax server"""
        self.server.serve_forever()

    def ping(self, *args):
        """just to test the server"""
        print("ping", args)
        self.client.send_message("/from_server", "pong")

    def send(self, address, content):
        """global method to send a message"""
        if self.debug:
            print('Sending following message')
            print(address)
            print(content)
        self.client.send_message(address, content)

    def print(self, *args):
        print(*args)
        self.send('/print', *args)


# OSC decorator
def osc_parse(func):
    """decorates a python function to automatically transform args and kwargs coming from Max"""

    def func_embedding(address, *args):
        t_args = tuple();
        kwargs = {}
        for a in args:
            if issubclass(type(a), str):
                if "=" in a:
                    key, value = a.split("=")
                    kwargs[key] = value
                else:
                    t_args = t_args + (a,)
            else:
                t_args = t_args + (a,)
        return func(*t_args, **kwargs)

    return func_embedding


def max_format(v):
    """Format some Python native types for Max"""
    if issubclass(type(v), (list, tuple)):
        if len(v) == 0:
            return ' "" '
        return ''.join(['%s ' % (i) for i in v])
    else:
        return v


def dict2str(dic):
    """Convert a python dict to a Max message filling a dict object"""
    str = ''
    for k, v in dic.items():
        str += ', set %s %s' % (k, max_format(v))
    return str[2:]


class TCP_UDP_interface(OSCServer):
    """
    Key class for the Flow synthesizer server.

    Example :
    >>> server = FlowServer(1234, 1235) # Creating server
    >>> server.run() # Running server
    """

    def __init__(self, **kwargs):
        self.osc_attributes = []
        # Latent paths variables
        super(TCP_UDP_interface, self).__init__(kwargs['in_port'], kwargs['out_port'], kwargs['ip'], )
        self.port_tcp = kwargs['port_tcp']
        self.ip_server = kwargs['ip_server']
        self.max_message = 8192

    def init_bindings(self, osc_attributes=[]):
        """ Set of OSC messages handled """
        super(TCP_UDP_interface, self).init_bindings(self.osc_attributes)
        self.dispatcher.map('/set_temperature', osc_parse(self.set_temperature))
        self.dispatcher.map('/load_piano_score', osc_parse(self.load_piano_score))
        self.dispatcher.map('/orchestrate', osc_parse(self.orchestrate))

    def interface_TCP_to_AWS(self, function_name, value):
        # Send data to AWS
        message = pickle.dumps(dict(
            value=value,
            function=function_name
        ))
        # Create a socket (SOCK_STREAM means a TCP socket)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Connect to server and send data
            sock.connect((self.ip_server, self.port_tcp))
            sock.sendall(message)

            # Load data (dynamic size)
            payload_size = struct.calcsize('L')
            data_rcv = b''
            while len(data_rcv) < payload_size:
                data_rcv += sock.recv(4096)
            packed_msg_size = data_rcv[:payload_size]
            data_rcv = data_rcv[payload_size:]
            msg_size = struct.unpack("L", packed_msg_size)[0]
            # Retrieve all data based on message size
            while len(data_rcv) < msg_size:
                data_rcv += sock.recv(4096)

            received = pickle.loads(data_rcv)

        # Send to UDP
        if received['function'] == 'piano_loaded':
            self.send('/piano_loaded', '0')
            sanity_check_from_server = received['value']
            return sanity_check_from_server
        elif received['function'] == 'orchestrate':
            max_length = received['value']['max_length']
            self.send(f'/init_orchestration', max_length)
            sanity_check_received = received['value']['sanity_check']
            formatted_output = received['value']['formatted_output']
            for _, list_formatted in formatted_output.items():
                if len(list_formatted) == 0:
                    continue
                self.send(f'/orchestration', list_formatted)
            sanity_check = int(sum([sum(v) for v in formatted_output.values()]))
            if sanity_check_received != sanity_check:
                print(f'#### Data lost: orchestration from AWS to local')
        elif received['function'] == 'nothing':
            pass

    def set_temperature(self, v):
        # Send to AWS
        print(f'Set temperature: {v}')
        self.interface_TCP_to_AWS('set_temperature', value=v)

    def load_piano_score(self, *v):
        """

        When a midi/xm file is dropped in the max/msp patch, it is send to this function.
        Reads the input file in the self.piano matrix
        """
        print(f'Sending piano score to server:...')
        if v == 'none':
            return
        sanity_check = sum([e for e in v if type(e) != str])
        sanity_check_from_server = self.interface_TCP_to_AWS('load_piano_score', value=v)
        if sanity_check_from_server != sanity_check:
            print(f'#### Data lost: piano clip from local to AWS')
        print(f'... piano score loaded')

    def orchestrate(self):
        print(f'Request orchestration to server...')
        self.interface_TCP_to_AWS('orchestrate', value=[])
        self.send('/orchestration_done', '0')
        print(f'...done!')
        return


def main(args):
    interface = TCP_UDP_interface(in_port=args['in_port_udp'],
                                  out_port=args['out_port_udp'],
                                  port_tcp=args['port_tcp'],
                                  ip='127.0.0.1',
                                  ip_server=args['ip_server'])
    print(f'[Interface to server {args["ip_server"]} on port {args["port_tcp"]}]')
    print(f'[Local communication with Max: in-port={args["in_port_udp"]}, out_port={args["out_port_udp"]}]')
    interface.run()


if __name__ == '__main__':
    args = dict(
        ip_server='63.33.36.17',
        port_tcp=5001,
        in_port_udp=5002,
        out_port_udp=5003
    )
    main(args=args)

    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--ip_server', type=str, default='63.33.36.17')
    # parser.add_argument('--port_tcp', type=int, default=5001)
    # parser.add_argument('--in_port_udp', type=int, default=5002)
    # parser.add_argument('--out_port_udp', type=int, default=5003)
    # args = parser.parse_args()
    # main(args=vars(args))

    # HOST, PORT = args.ip_server, args.port_tcp
    # np_array = np.random.rand(3, 2)
    # data = dict(
    #     np_array=np_array,
    #     function='orchestrate'
    # )
    # message = pickle.dumps(data)
    # # Create a socket (SOCK_STREAM means a TCP socket)
    # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    #     # Connect to server and send data
    #     sock.connect((HOST, PORT))
    #     sock.sendall(message)
    #     # Receive data from the server and shut down
    #     received = str(sock.recv(1024), "utf-8")
    #
    # print("Sent:     {}".format(data))
    # print("Received: {}".format(received))
