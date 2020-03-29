import logging
from logging import handlers as logging_handlers

import click
import flask
import music21
import numpy as np
import torch
from DatasetManager.arrangement.arrangement_dataset import ArrangementDataset
from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata
from flask import Flask, request
from flask_cors import CORS
from music21 import musicxml, metadata

from Transformer.arrangement.arrangement_data_processor import ArrangementDataProcessor
from Transformer.transformer import Transformer

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = './uploads'
ALLOWED_EXTENSIONS = {'midi'}

# INITIALIZATION
xml_response_headers = {"Content-Type": "text/xml",
                        "charset": "utf-8"
                        }
mp3_response_headers = {"Content-Type": "audio/mpeg3"}

model = None
# Generation parameters
_subdivision = None
_batch_size = None
_banned_instruments = None
_temperature = None
_context_size = None
# Piano score
_piano = None
# Orchestra init (before even the first pass of the model, i.e. filled with MASK and REST symbols
_orchestra_init = None
_orchestra_silenced_instruments = None

# TODO use this parameter or extract it from the metadata somehow
timesignature = music21.meter.TimeSignature('4/4')

# generation parameters
# todo put in click?
batch_size_per_voice = 8

metadatas = [
    FermataMetadata(),
    TickMetadata(subdivision=_subdivision),
    KeyMetadata()
]


# def get_fermatas_tensor(metadata_tensor: torch.Tensor) -> torch.Tensor:
#     """
#     Extract the fermatas tensor from a metadata tensor
#     """
#     fermatas_index = [m.__class__ for m in metadatas].index(
#         FermataMetadata().__class__)
#     # fermatas are shared across all voices so we only consider the first voice
#     soprano_voice_metadata = metadata_tensor[0]
#
#     # `soprano_voice_metadata` has shape
#     # `(sequence_duration, len(metadatas + 1))`  (accouting for the voice
#     # index metadata)
#     # Extract fermatas for all steps
#     return soprano_voice_metadata[:, fermatas_index]


@click.command()
# Model (high-level)
@click.option('--block_attention', is_flag=True,
              help='Do we use block attention ?')
@click.option('--hierarchical', is_flag=True,
              help='Do we connect encoder and decoder in a hierarchical way')
@click.option('--nade', is_flag=True,
              help='Orderless auto-regressive model')
# Model (low-level)
@click.option('--num_layers', default=2,
              help='number of layers of the LSTMs')
@click.option('--dropout', default=0.1,
              help='amount of dropout between layers')
@click.option('--input_dropout', default=0.2,
              help='amount of dropout on input')
@click.option('--per_head_dim', default=64,
              help='Feature dimension in each head')
@click.option('--num_heads', default=8,
              help='Number of heads')
@click.option('--local_position_embedding_dim', default=8,
              help='Embedding size for local positions')
@click.option('--position_ff_dim', default=1024,
              help='Hidden dimension of the position-wise ffnn')
# Dataset
@click.option('--subdivision', default=2, type=int,
              help='subdivisions of qaurter note in dataset')
@click.option('--sequence_size', default=3, type=int,
              help='length of piano chunks')
@click.option('--velocity_quantization', default=2, type=int,
              help='number of possible velocities')
@click.option('--max_transposition', default=12, type=int,
              help='maximum pitch shift allowed when transposing for data augmentation')
#  Generation
@click.option('--suffix', default="", type=str,
              help='suffix for model name')
# Server
@click.option('--port', default=5000,
              help='Server listen to this port')
def init_app(block_attention,
             hierarchical,
             nade,
             num_layers,
             dropout,
             input_dropout,
             per_head_dim,
             num_heads,
             local_position_embedding_dim,
             position_ff_dim,
             suffix,
             subdivision,
             sequence_size,
             velocity_quantization,
             max_transposition,
             port
             ):
    global metadatas
    global _subdivision
    global _batch_size
    global _banned_instruments
    global _temperature
    global _lowest_entropy_first
    global _context_size

    _subdivision = subdivision
    _batch_size = 1
    _banned_instruments = []
    _temperature = 1.2
    _lowest_entropy_first = True

    gpu_ids = [int(gpu) for gpu in range(torch.cuda.device_count())]
    print(gpu_ids)

    dataset_manager = DatasetManager()
    arrangement_dataset_kwargs = {
        'transpose_to_sounding_pitch': True,
        'subdivision': subdivision,
        'sequence_size': sequence_size,
        'velocity_quantization': velocity_quantization,
        'max_transposition': max_transposition,
        'compute_statistics_flag': False
    }
    dataset: ArrangementDataset = dataset_manager.get_dataset(
        name='arrangement',
        **arrangement_dataset_kwargs
    )

    reducer_input_dim = num_heads * per_head_dim

    processor_encoder = ArrangementDataProcessor(dataset=dataset,
                                                 embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                 reducer_input_dim=reducer_input_dim,
                                                 local_position_embedding_dim=local_position_embedding_dim,
                                                 flag_orchestra=False,
                                                 block_attention=False)

    processor_decoder = ArrangementDataProcessor(dataset=dataset,
                                                 embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                 reducer_input_dim=reducer_input_dim,
                                                 local_position_embedding_dim=local_position_embedding_dim,
                                                 flag_orchestra=True,
                                                 block_attention=block_attention)

    _context_size = processor_decoder.num_frames_orchestra - 1

    global model
    model = Transformer(dataset=dataset,
                        data_processor_encoder=processor_encoder,
                        data_processor_decoder=processor_decoder,
                        num_heads=num_heads,
                        per_head_dim=per_head_dim,
                        position_ff_dim=position_ff_dim,
                        hierarchical_encoding=hierarchical,
                        block_attention=block_attention,
                        nade=nade,
                        num_layers=num_layers,
                        dropout=dropout,
                        input_dropout=input_dropout,
                        conditioning=True,
                        lr=0,
                        gpu_ids=gpu_ids,
                        suffix=suffix,
                        )

    model.load_overfit()
    model.cuda()

    # TODO : piano should be modifiable (by dropping mxml file ?)
    filepath = "/home/leo/Recherche/Databases/Orchestration/arrangement_mxml/source_for_generation/chopin_Prel_Op28_20_xs.xml"
    global _piano
    global _rhythm_piano
    global _orchestra
    global _orchestra_silenced_instruments
    _piano, _rhythm_piano, _orchestra, _orchestra_silenced_instruments = \
        model.data_processor_encoder.init_generation_filepath(_batch_size, filepath,
                                                              banned_instruments=_banned_instruments,
                                                              subdivision=_subdivision)

    # launch the script
    # use threaded=True to fix Chrome/Chromium engine hanging on requests
    # [https://stackoverflow.com/a/30670626]
    local_only = False
    if local_only:
        # accessible only locally:
        app.run(threaded=True)
    else:
        # accessible from outside:
        app.run(host='0.0.0.0', port=port, threaded=True)


@app.route('/generate', methods=['GET', 'POST'])
def compose():
    """
    Return a new, generated sheet
    Usage:
        - Request: empty, generation is done in an unconstrained fashion
        - Response: a sheet, MusicXML
    """
    global model
    global _subdivision
    global _piano
    global _orchestra
    global _orchestra_silenced_instruments
    global _batch_size
    global _temperature
    global _context_size
    global _lowest_entropy_first

    # TODO Make these parameters which can be modified through the interface

    _orchestra = model.generation_arrangement_entropy_based_ordering(
        piano=_piano,
        orchestra_init=_orchestra,
        orchestra_silenced_instruments=_orchestra_silenced_instruments,
        temperature=_temperature,
        batch_size=_batch_size,
        lowest_entropy_first=_lowest_entropy_first,
        plot_attentions=False
    )

    # Orchestra to music21 stream
    durations_piano = np.asarray(list(_rhythm_piano[1:]) + [_subdivision]) - np.asarray(list(_rhythm_piano[:-1]) + [0])
    orchestra_flat_cpu = _orchestra[:, _context_size:-_context_size].cpu()
    orchestra_stream = model.dataset.orchestra_tensor_to_score(orchestra_flat_cpu[0], durations_piano)

    # orchestra_stream = music21.corpus.chorales.getByTitle("Sach' Gott heimgestellt")
    response = sheet_to_json_response(orchestra_stream)

    return response


@app.route('/test-generate', methods=['GET'])
def ex():
    _current_sheet = next(music21.corpus.chorales.Iterator())
    return sheet_to_xml_response(_current_sheet)


@app.route('/musicxml-to-midi', methods=['POST'])
def get_midi():
    """
    Convert the provided MusicXML sheet to MIDI and return it
    Usage:
        POST -d @sheet.mxml /musicxml-to-midi
        - Request: the payload is expected to contain the sheet to convert, in
        MusicXML format
        - Response: a MIDI file
    """
    sheetString = request.data
    sheet = music21.converter.parseData(sheetString, format="musicxml")
    insert_musicxml_metadata(sheet)

    return sheet_to_midi_response(sheet)


@app.route('/timerange-change', methods=['POST'])
def timerange_change():
    """
    Perform local re-generation on a sheet and return the updated sheet
    Usage:
        POST /timerange-change?time_range_start_beat=XXX&time_range_end_beat=XXX
        - Request:
            The payload is expected to be a JSON with the following keys:
                * 'sheet': a string containing the sheet to modify, in MusicXML
                  format
                * 'fermatas': a list of integers describing the positions of
                  fermatas in the sheet
                  TODO: could store the fermatas in the MusicXML client-side
            The start and end positions (in beats) of the portion to regenerate
            are passed as arguments in the URL:
                * time_range_start_quarter, integer:
        - Response:
            A JSON document with same schema as the request containing the
            updated sheet and fermatas
    """
    global model
    global _orchestra
    global _temperature
    global _batch_size
    global _rhythm_piano
    global _context_size
    global _lowest_entropy_first
    request_parameters = parse_timerange_request(request)
    time_range_start_quarter = request_parameters['time_range_start_quarter']
    time_range_end_quarter = request_parameters['time_range_end_quarter']

    time_events = list_events_from_quarter_range(time_range_start_quarter, time_range_end_quarter)

    _orchestra = model.generation_arrangement_entropy_based_ordering(
        piano=_piano,
        orchestra_init=_orchestra,
        orchestra_silenced_instruments=_orchestra_silenced_instruments,
        temperature=_temperature,
        batch_size=_batch_size,
        lowest_entropy_first=_lowest_entropy_first,
        plot_attentions=False,
        events=time_events
    )

    # Extract only the new part


    # Orchestra to music21 stream
    durations_piano = np.asarray(list(_rhythm_piano[1:]) + [_subdivision]) - np.asarray(list(_rhythm_piano[:-1]) + [0])
    orchestra_flat_cpu = _orchestra[:, _context_size:-_context_size].cpu()
    orchestra_stream = model.dataset.orchestra_tensor_to_score(orchestra_flat_cpu[0], durations_piano)

    response = sheet_to_json_response(orchestra_stream)

    return response


@app.route('/analyze-notes', methods=['POST'])
def dummy_read_audio_file():
    global deepbach
    import wave
    print(request.args)
    print(request.files)
    chunk = 1024
    audio_fp = wave.open(request.files['audio'], 'rb')
    data = audio_fp.readframes(chunk)
    print(data)
    notes = ['C', 'D', 'Toto', 'Tata']

    return flask.jsonify({'success': True, 'notes': notes})


def list_events_from_quarter_range(start_quarter, end_quarter):
    global _subdivision
    global _rhythm_piano
    global context_size
    events = [index+context_size for index, time_quarter in enumerate(_rhythm_piano)
              if (time_quarter >= start_quarter*_subdivision) and (time_quarter < end_quarter*_subdivision)]
    return events

def insert_musicxml_metadata(sheet: music21.stream.Stream):
    """
    Insert various metadata into the provided XML document
    The timesignature in particular is required for proper MIDI conversion
    """
    global timesignature

    # from music21.clef import TrebleClef, BassClef, Treble8vbClef
    # for part, name, clef in zip(
    #         sheet.parts,
    #         ['soprano', 'alto', 'tenor', 'bass'],
    #         [TrebleClef(), TrebleClef(), Treble8vbClef(), BassClef()]
    # ):
    #     # empty_part = part.template()
    #     part.insert(0, timesignature)
    #     part.insert(0, clef)
    #     part.id = name
    #     part.partName = name

    md = metadata.Metadata()
    sheet.insert(0, md)
    # required for proper musicXML formatting
    sheet.metadata.title = 'YOYO'
    sheet.metadata.composer = 'YOYO'
    return


def parse_timerange_request(request):
    """
    must cast
    :param req:
    :return:
    """
    json_data = request.get_json(force=True)
    time_range_start_quarter = int(request.args.get('time_range_start_quarter'))
    time_range_end_quarter = int(request.args.get('time_range_end_quarter'))

    sheet = music21.converter.parseData(json_data['sheet'], format="musicxml")

    return {
        'sheet': sheet,
        'time_range_start_quarter': time_range_start_quarter,
        'time_range_end_quarter': time_range_end_quarter,
        # 'fermatas_tensor': fermatas_tensor
    }


def sheet_to_xml_bytes(sheet: music21.stream.Stream):
    """Convert a music21 sheet to a MusicXML document"""
    # first insert necessary MusicXML metadata
    insert_musicxml_metadata(sheet)

    sheet_to_xml_bytes = musicxml.m21ToXml.GeneralObjectExporter(sheet).parse()

    return sheet_to_xml_bytes


def sheet_to_xml_response(sheet: music21.stream.Stream):
    """Generate and send XML sheet"""
    xml_sheet_bytes = sheet_to_xml_bytes(sheet)

    response = flask.make_response((xml_sheet_bytes, xml_response_headers))
    return response


def sheet_to_json_response(sheet: music21.stream.Stream):
    sheet_xml_string = sheet_to_xml_bytes(sheet).decode('utf-8')
    fermatas_list = []

    return flask.jsonify({
        'sheet': sheet_xml_string,
        'fermatas': fermatas_list
    })


def sheet_to_midi_response(sheet):
    """
    Convert the provided sheet to midi and send it as a file
    """
    midiFile = sheet.write('midi')
    return flask.send_file(midiFile,
                           mimetype="audio/midi",
                           cache_timeout=-1  # disable cache
                           )


# def sheet_to_mp3_response(sheet):
#     """Generate and send MP3 file
#     Uses server-side `timidity`
#     """
#     sheet.write('midi', fp='./uploads/midi.mid')
#     os.system(f'rm uploads/midi.mp3')
#     os.system(f'timidity uploads/midi.mid -Ow -o - | '
#               f'ffmpeg -i - -acodec libmp3lame -ab 64k '
#               f'uploads/midi.mp3')
#     return flask.send_file('uploads/midi.mp3')


if __name__ == '__main__':
    file_handler = logging_handlers.RotatingFileHandler(
        'app.log', maxBytes=10000, backupCount=5)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)

init_app()
