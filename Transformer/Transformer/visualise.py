import os
import shutil
import matplotlib.pyplot as plt
from DatasetManager.helpers import TIME_SHIFT, STOP_SYMBOL

plt.style.use('seaborn-white')


def visualize_arrangement(
        model,
        batch_size,
        log_dir
):
    (_, _, generator_test) = model.dataset.data_loaders(batch_size=batch_size)

    monitored_quantities_val = model.epoch(
        data_loader=generator_test,
        epoch_id=0,
        train=False,
        num_batches=1,
        label_smoothing=0,
        loss_on_last_frame=False,
        return_attns=True
    )

    # Plot attentions
    attention_plot_dir = f'{log_dir}/figures/attentions'
    if os.path.isdir(attention_plot_dir):
        shutil.rmtree(attention_plot_dir)
    os.makedirs(attention_plot_dir)

    attentions = monitored_quantities_val['attentions']
    encoder_self_attention = attentions['encoder_self']
    decoder_self_attention = attentions['decoder_self']
    encoder_decoder_attention = attentions['encoder_decoder']

    piano_input = monitored_quantities_val['piano_input']
    orchestra_input = monitored_quantities_val['orchestra_input']
    orchestra_output = monitored_quantities_val['orchestra_output']

    #  Write corresponding input and output in text format
    for batch_ind in range(batch_size):
        batch_plot_dir = f'{attention_plot_dir}/{batch_ind}'
        if not os.path.isdir(batch_plot_dir):
            os.mkdir(batch_plot_dir)
        writefile = f'{batch_plot_dir}/data.txt'
        open(writefile, 'w').close()
        if model.data_processor_encoder.name == 'arrangement_voice_data_processor':
            seq2text_voice(piano_input[batch_ind], model.dataset.index2midi_pitch_piano, writefile,
                           orchestra_flag=False, name='piano_input')
        elif model.data_processor_encoder.name == 'arrangement_midiPiano_data_processor':
            seq2text_midi(piano_input[batch_ind], model.dataset.index2symbol_piano, writefile, name='piano_input')
        seq2text_voice(orchestra_input[batch_ind], model.dataset.index2midi_pitch, writefile, orchestra_flag=True, name='orchestra_input')
        seq2text_voice(orchestra_output[batch_ind], model.dataset.index2midi_pitch, writefile, orchestra_flag=True, name='orchestra_output')

    num_layers = len(encoder_decoder_attention)
    batch_size_times_heads = len(encoder_decoder_attention[0])
    num_heads = batch_size_times_heads // batch_size
    for layer_ind in range(num_layers):
        # View to separate heads
        dims_enc_dec = encoder_decoder_attention[layer_ind].shape
        enc_dec_layer = encoder_decoder_attention[layer_ind].view(batch_size, num_heads, dims_enc_dec[1],
                                                                  dims_enc_dec[2])
        dims_enc_self = encoder_self_attention[layer_ind].shape
        enc_self_layer = encoder_self_attention[layer_ind].view(batch_size, num_heads, dims_enc_self[1],
                                                                dims_enc_self[2])
        dims_dec_self = decoder_self_attention[layer_ind].shape
        dec_self_layer = decoder_self_attention[layer_ind].view(batch_size, num_heads, dims_dec_self[1],
                                                                dims_dec_self[2])

        for batch_ind in range(batch_size):
            batch_plot_dir = f'{attention_plot_dir}/{batch_ind}'
            savepath = f'{batch_plot_dir}/enc_dec_{layer_ind}.pdf'
            subplot_attentions(enc_dec_layer[batch_ind], savepath)
            savepath = f'{batch_plot_dir}/enc_self_{layer_ind}.pdf'
            subplot_attentions(enc_self_layer[batch_ind], savepath)
            savepath = f'{batch_plot_dir}/dec_self_{layer_ind}.pdf'
            subplot_attentions(dec_self_layer[batch_ind], savepath)


def seq2text_midi(seq, index2symbol, writefile, name):
    symbols = []
    for message in seq:
        symbol = index2symbol[message]
        symbols.append(symbol)
        if symbol in [TIME_SHIFT, STOP_SYMBOL]:
            symbols.append('\n')

    with open(writefile, 'a') as ff:
        ff.write(f'{name} \n')
        for sym in symbols:
            if sym == '\n':
                ff.write(f'{sym}')
            else:
                ff.write(f'{sym} ')
        ff.write(f'\n\n')
    return symbols

def seq2text_voice(seq, index2symbol, writefile, orchestra_flag, name):
    time_frames, num_voices = seq.shape
    symbols = []
    for t in range(time_frames):
        for voice in range(num_voices):
            index = seq[t, voice]
            if orchestra_flag:
                symbols.append(index2symbol[voice][index])
            else:
                symbols.append(index2symbol[index])
        symbols.append('\n')
    with open(writefile, 'a') as ff:
        ff.write(f'{name} \n')
        for sym in symbols:
            if sym == '\n':
                ff.write(f'{sym}')
            elif sym == 'rest':
                ff.write(f'R  ')
            # elif sym == '__':
            #     ff.write(f'S  ')
            else:
                ff.write(f'{sym} ')
        ff.write(f'\n\n')
    return symbols


def subplot_attentions(attention, savepath):
    num_heads = len(attention)
    plt.clf()
    # equivalent but more general
    fig, axs = plt.subplots(num_heads // 2, 2)
    for head_ind in range(num_heads):
        x_ind = head_ind // 2
        y_ind = head_ind % 2
        axs[x_ind, y_ind].imshow(attention[head_ind].numpy())
        axs[x_ind, y_ind].set_title(f'h_{head_ind}')
        axs[x_ind, y_ind].set_xlabel('output')
        axs[x_ind, y_ind].set_ylabel('input')
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    plt.savefig(savepath)
    plt.close()
