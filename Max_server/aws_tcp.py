import argparse
import os
import pickle
import socketserver

import dataset_import
import torch
from DatasetManager.dataset_manager import DatasetManager
from Transformer.transformer import Transformer


def main(args):
    """

    :param args:
    :return:
    """
    dropout = 0.
    input_dropout = 0.
    input_dropout_token = 0.
    mixup = False
    scheduled_training = 0.
    group_instrument_per_section = False
    reduction_flag = False
    lr = 1.
    cpc_config_name = None
    subdivision = args.subdivision

    # Use all gpus available
    gpu_ids = [int(gpu) for gpu in range(torch.cuda.device_count())]
    print(f'Using GPUs {gpu_ids}')
    if len(gpu_ids) == 0:
        device = 'cpu'
    else:
        device = 'cuda'

    # Get dataset
    dataset_manager = DatasetManager()
    dataset, processor_decoder, processor_encoder, processor_encodencoder = \
        dataset_import.get_dataset(dataset_manager, args.dataset_type, args.subdivision, args.sequence_size,
                                   args.velocity_quantization, args.max_transposition,
                                   args.num_heads, args.per_head_dim, args.local_position_embedding_dim,
                                   args.block_attention,
                                   group_instrument_per_section, args.nade, cpc_config_name, args.double_conditioning,
                                   args.instrument_presence_in_encoder)

    # Load model
    model = Transformer(dataset=dataset,
                        data_processor_encodencoder=processor_encodencoder,
                        data_processor_encoder=processor_encoder,
                        data_processor_decoder=processor_decoder,
                        num_heads=args.num_heads,
                        per_head_dim=args.per_head_dim,
                        position_ff_dim=args.position_ff_dim,
                        enc_dec_conditioning=args.enc_dec_conditioning,
                        hierarchical_encoding=args.hierarchical,
                        block_attention=args.block_attention,
                        nade=args.nade,
                        conditioning=args.conditioning,
                        double_conditioning=args.double_conditioning,
                        num_layers=args.num_layers,
                        dropout=dropout,
                        input_dropout=input_dropout,
                        input_dropout_token=input_dropout_token,
                        lr=lr, reduction_flag=reduction_flag,
                        gpu_ids=gpu_ids,
                        suffix=args.suffix,
                        mixup=mixup,
                        scheduled_training=scheduled_training
                        )

    model.load_overfit(device=device)
    model.to(device)
    model = model.eval()

    # Dir for writing generated files
    writing_dir = f'{os.getcwd()}/generation'
    if not os.path.isdir(writing_dir):
        os.makedirs(writing_dir)

    # Create server
    server_address = ('127.0.0.1', args.port)
    server = OrchestraServer(server_address, model, subdivision, writing_dir)
    server.serve_forever()


class _TCPHandler(socketserver.BaseRequestHandler):
    """Handles correct TCP messages for all types of server."""
    def handle(self) -> None:
        """Calls the handlers via dispatcher

        This method is called after a basic sanity check was done on the datagram,
        whether this datagram looks like an osc message or bundle.
        If not the server won't call it and so no new
        threads/processes will be spawned.
        """
        # Get data
        self.data = pickle.loads(self.request.recv(8192).strip())
        # Process
        print(self.data['np_array'])
        # Return
        self.request.sendall(bytes("AUREVOIR\n", "utf-8"))


class OrchestraServer(socketserver.TCPServer):
    def __init__(self, server_address, model, subdivision, writing_dir):
        # server
        super().__init__(server_address, _TCPHandler)

        # loacl computation
        self._model = model
        self.writing_dir = writing_dir
        self.subdivision = subdivision
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

    # def set_temperature(self, v):
    #     self.temperature = v
    #     print(v)
    #
    # def load_piano_score(self, *v):
    #     """
    #
    #     When a midi/xm file is dropped in the max/msp patch, it is send to this function.
    #     Reads the input file in the self.piano matrix
    #     """
    #     # Remove prepended shit
    #     if v == 'none':
    #         return
    #
    #     length = math.ceil(v[-1] * self.subdivision)
    #     if length < 1:
    #         return
    #
    #     # Remove prefix and suffix useless messages
    #     v = v[2:-2]
    #
    #     # List to pianoroll
    #     pianoroll = np.zeros((length, 128))
    #     onsets = np.zeros((length, 128))
    #     pitch = None
    #     start_t = None
    #     duration = None
    #     velocity = None
    #     for counter, message in enumerate(v):
    #         if counter % 6 == 0:
    #             continue
    #         elif counter % 6 == 1:
    #             pitch = message
    #         elif counter % 6 == 2:
    #             start_t = int(message * self.subdivision)
    #         elif counter % 6 == 3:
    #             duration = int(message * self.subdivision)
    #         elif counter % 6 == 4:
    #             velocity = message
    #         elif counter % 6 == 5:
    #             pianoroll[start_t:start_t + duration, pitch] = 100
    #             onsets[start_t, pitch] = 100
    #
    #     piano, _, rhythm_piano, orchestra_init, \
    #     instruments_presence, orchestra_silenced_instruments, orchestra_unknown_instruments = \
    #         self._model.data_processor_decoder.dataset.pianoroll_to_formated_tensor(
    #             pianoroll_piano={'Piano': pianoroll},
    #             onsets_piano={'Piano': onsets},
    #             batch_size=1,
    #             context_length=self.context_size,
    #             banned_instruments=self.banned_instruments,
    #             unknown_instruments=self.unknown_instruments
    #         )
    #
    #     self.piano = piano
    #     self.durations_piano = np.asarray(list(rhythm_piano[1:]) + [self.subdivision]) - np.asarray(
    #         list(rhythm_piano[:-1]) + [0])
    #     self.orchestra_init = orchestra_init
    #     self.instrument_presence = instruments_presence
    #     self.orchestra_silenced_instruments = orchestra_silenced_instruments
    #     self.orchestra_unknown_instruments = orchestra_unknown_instruments
    #
    #     print('piano score loaded!')
    #     self.send('/piano_loaded', '0')
    #     return
    #
    # def orchestrate(self):
    #     if self.piano is None:
    #         print('No piano score has been inputted :(')
    #         return
    #
    #     print('orchestrating...')
    #     print(f'T={self.temperature}')
    #
    #     orchestra = generation_arrangement(
    #         model=self._model,
    #         piano=self.piano,
    #         orchestra_init=self.orchestra_init,
    #         orchestra_silenced_instruments=self.orchestra_silenced_instruments,
    #         instruments_presence=self.instrument_presence,
    #         temperature=self.temperature,
    #         batch_size=1,
    #         number_sampling_steps=1
    #     )
    #
    #     # Write each instrument in the orchestration as a separate xml file
    #     orchestra_writing = orchestra[0, self.context_size:-self.context_size]
    #
    #     _, _, score_dict = self._model.dataset.orchestra_tensor_to_score(
    #         tensor_score=orchestra_writing,
    #         format='midi',
    #         durations=self.durations_piano,
    #         subdivision=self.subdivision)
    #
    #     # For Ableton
    #     #  First get longest clip and send init_orchestra to max
    #     max_length = 0
    #     for _, list in score_dict.items():
    #         if len(list) == 0:
    #             continue
    #         elem = list[-1]
    #         max_length = max(max_length, elem[1] + elem[2])
    #     if max_length == 0:
    #         return
    #     self.send(f'/init_orchestration', max_length / self.subdivision)
    #
    #     # Then send the content
    #     for instrument_name, list in score_dict.items():
    #         if len(list) == 0:
    #             continue
    #         list_formatted = [self.instrument_to_index[instrument_name]]
    #         for elem in list:
    #             list_formatted.append(elem[0])
    #             list_formatted.append(elem[1] / self.subdivision)  # start
    #             list_formatted.append(elem[2] / self.subdivision)  # duration
    #         self.send(f'/orchestration', list_formatted)
    #
    #     print('done')
    #     self.send('/orchestration_done', '0')
    #     return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5001)
    # Model arguments
    parser.add_argument('--hierarchical', type=bool, default=False)
    parser.add_argument('--nade', type=bool, default=False)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--per_head_dim', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--local_position_embedding_dim', type=int, default=8)
    parser.add_argument('--position_ff_dim', type=int, default=2048)
    parser.add_argument('--enc_dec_conditioning', type=str, default='split')
    parser.add_argument('--dataset_type', type=str, default='arrangement_voice')
    parser.add_argument('--conditioning', type=bool, default=True)
    parser.add_argument('--double_conditioning', type=str, default=None)
    parser.add_argument('--subdivision', type=int, default=16)
    parser.add_argument('--sequence_size', type=int, default=7)
    parser.add_argument('--velocity_quantization', type=int, default=2)
    parser.add_argument('--max_transposition', type=int, default=12)
    parser.add_argument('--instrument_presence_in_encoder', type=bool, default=False)
    parser.add_argument('--block_attention', type=bool, default=False)
    parser.add_argument('--suffix', type=str, default='REFERENCE')
    args = parser.parse_args()
    main(args)
