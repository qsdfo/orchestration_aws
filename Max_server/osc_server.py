import math
import re
import numpy as np
from Transformer.generate import generation_arrangement
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client

from collections import Iterable
import socket

from typing import Union
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_message import OscMessage
from pythonosc.osc_bundle import OscBundle


# Helper function to parse attribute
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
    def __init__(self, in_port, out_port, ip, ip_client, *args):
        super(OSCServer, self).__init__()
        # OSC library objects
        self.dispatcher = dispatcher.Dispatcher()
        self.client = SimpleUDPClientCustom(ip_client, out_port)
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


class OrchestraServer(OSCServer):
    """
    Key class for the Flow synthesizer server.

    Example :
    >>> server = FlowServer(1234, 1235) # Creating server
    >>> server.run() # Running server
    """

    def __init__(self, *args, **kwargs):
        self._model = kwargs.get('model')
        self.writing_dir = kwargs.get('writing_dir')
        # Interface variables
        self.subdivision = kwargs.get('subdivision')
        self.temperature = 1.2
        self.piano = None
        self.orchestra_init = None
        self.orchestra = None
        self.durations_piano = None
        self.instrument_presence = None
        self.orchestra_silenced_instruments = None
        self.orchestra_unknown_instruments = None
        self.banned_instruments = []
        self.unknown_instruments = []
        self.context_size = self._model.data_processor_decoder.num_frames_orchestra - 1

        self.instrument_to_index = {
            'Piccolo': 1,
            'Flute': 2,
            'Oboe': 3,
            'Clarinet': 4,
            'Bassoon': 5,
            'Horn': 6,
            'Trumpet': 7,
            'Trombone': 8,
            'Tuba': 9,
            'Violin_1': 10,
            'Violin_2': 11,
            'Viola': 12,
            'Violoncello': 13,
            'Contrabass': 14
        }

        self.osc_attributes = []
        # Latent paths variables
        super(OrchestraServer, self).__init__(*args)
        self.debug = kwargs.get('debug')

    def init_bindings(self, osc_attributes=[]):
        """ Set of OSC messages handled """
        super(OrchestraServer, self).init_bindings(self.osc_attributes)
        self.dispatcher.map('/set_temperature', osc_parse(self.set_temperature))
        self.dispatcher.map('/load_piano_score', osc_parse(self.load_piano_score))
        self.dispatcher.map('/orchestrate', osc_parse(self.orchestrate))

    def set_temperature(self, v):
        self.temperature = v
        print(v)

    def load_piano_score(self, *v):
        """

        When a midi/xm file is dropped in the max/msp patch, it is send to this function.
        Reads the input file in the self.piano matrix
        """
        # Remove prepended shit
        if v == 'none':
            return

        length = math.ceil(v[-1] * self.subdivision)
        if length < 1:
            return

        # Remove prefix and suffix useless messages
        v = v[2:-2]

        # List to pianoroll
        pianoroll = np.zeros((length, 128))
        onsets = np.zeros((length, 128))
        pitch = None
        start_t = None
        duration = None
        velocity = None
        for counter, message in enumerate(v):
            if counter % 6 == 0:
                continue
            elif counter % 6 == 1:
                pitch = message
            elif counter % 6 == 2:
                start_t = int(message * self.subdivision)
            elif counter % 6 == 3:
                duration = int(message * self.subdivision)
            elif counter % 6 == 4:
                velocity = message
            elif counter % 6 == 5:
                pianoroll[start_t:start_t + duration, pitch] = 100
                onsets[start_t, pitch] = 100

        piano, _, rhythm_piano, orchestra_init, \
        instruments_presence, orchestra_silenced_instruments, orchestra_unknown_instruments = \
            self._model.data_processor_decoder.dataset.pianoroll_to_formated_tensor(
                pianoroll_piano={'Piano': pianoroll},
                onsets_piano={'Piano': onsets},
                batch_size=1,
                context_length=self.context_size,
                banned_instruments=self.banned_instruments,
                unknown_instruments=self.unknown_instruments
            )

        self.piano = piano
        self.durations_piano = np.asarray(list(rhythm_piano[1:]) + [self.subdivision]) - np.asarray(
            list(rhythm_piano[:-1]) + [0])
        self.orchestra_init = orchestra_init
        self.instrument_presence = instruments_presence
        self.orchestra_silenced_instruments = orchestra_silenced_instruments
        self.orchestra_unknown_instruments = orchestra_unknown_instruments

        print('piano score loaded!')
        self.send('/piano_loaded', '0')
        return

    def orchestrate(self):
        if self.piano is None:
            print('No piano score has been inputted :(')
            return

        print('orchestrating...')
        print(f'T={self.temperature}')

        orchestra = generation_arrangement(
            model=self._model,
            piano=self.piano,
            orchestra_init=self.orchestra_init,
            orchestra_silenced_instruments=self.orchestra_silenced_instruments,
            instruments_presence=self.instrument_presence,
            temperature=self.temperature,
            batch_size=1,
            number_sampling_steps=1
        )

        # Write each instrument in the orchestration as a separate xml file
        orchestra_writing = orchestra[0, self.context_size:-self.context_size]

        _, _, score_dict = self._model.dataset.orchestra_tensor_to_score(
            tensor_score=orchestra_writing,
            format='midi',
            durations=self.durations_piano,
            subdivision=self.subdivision)

        # For Ableton
        #  First get longest clip and send init_orchestra to max
        max_length = 0
        for _, list in score_dict.items():
            if len(list) == 0:
                continue
            elem = list[-1]
            max_length = max(max_length, elem[1] + elem[2])
        if max_length == 0:
            return
        self.send(f'/init_orchestration', max_length / self.subdivision)

        # Then send the content
        for instrument_name, list in score_dict.items():
            if len(list) == 0:
                continue
            list_formatted = [self.instrument_to_index[instrument_name]]
            for elem in list:
                list_formatted.append(elem[0])
                list_formatted.append(elem[1] / self.subdivision)  # start
                list_formatted.append(elem[2] / self.subdivision)  # duration
            self.send(f'/orchestration', list_formatted)

        print('done')
        self.send('/orchestration_done', '0')
        return


class UDPClientCustom(object):
    """OSC client to send :class:`OscMessage` or :class:`OscBundle` via UDP"""

    def __init__(self, address: str, port: int, allow_broadcast: bool = True) -> None:
        """Initialize client

        As this is UDP it will not actually make any attempt to connect to the
        given server at ip:port until the send() method is called.

        Args:
            address: IP address of server
            port: Port of server
            allow_broadcast: Allow for broadcast transmissions
        """
        for addr in socket.getaddrinfo(address, port, type=socket.SOCK_DGRAM):
            af, socktype, protocol, canonname, sa = addr

            try:
                self._sock = socket.socket(af, socktype)
            except OSError:
                continue
            break

        self._sock.setblocking(0)
        if allow_broadcast:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._address = address
        self._port = port

    def send(self, content: Union[OscMessage, OscBundle]) -> None:
        """Sends an :class:`OscMessage` or :class:`OscBundle` via UDP

        Args:
            content: Message or bundle to be sent
        """
        self._sock.sendto(content.dgram, (self._address, self._port))


class SimpleUDPClientCustom(UDPClientCustom):
    """Simple OSC client that automatically builds :class:`OscMessage` from arguments"""

    def send_message(self, address: str, value: Union[int, float, bytes, str, bool, tuple, list]) -> None:
        """Build :class:`OscMessage` from arguments and send to server

        Args:
            address: OSC address the message shall go to
            value: One or more arguments to be added to the message
        """
        builder = OscMessageBuilder(address=address)
        if value is None:
            values = []
        elif not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
            values = [value]
        else:
            values = value
        for val in values:
            builder.add_arg(val)
        msg = builder.build()
        self.send(msg)
