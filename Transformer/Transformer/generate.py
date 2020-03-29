import random

import numpy as np
import torch
import torch.nn.functional as F
from DatasetManager.helpers import START_SYMBOL, END_SYMBOL, PAD_SYMBOL, NO_SYMBOL, UNKNOWN_SYMBOL, YES_SYMBOL

from Transformer.helpers import to_numpy


def generation_arrangement(model,
                           piano,
                           orchestra_init,
                           orchestra_silenced_instruments,
                           instruments_presence,
                           temperature,
                           batch_size,
                           number_sampling_steps
                           ):
    model.eval()

    cpc_flag = hasattr(model.data_processor_decoder, 'cpc_model')
    if cpc_flag:
        raise NotImplementedError

    context_size = model.data_processor_decoder.num_frames_orchestra - 1
    number_piano_frames_to_orchestrate = piano.size()[1]

    with torch.no_grad():

        events = model.data_processor_encoder.get_range_generation(context_size, number_piano_frames_to_orchestrate)
        orchestra = orchestra_init

        for frame_index in events:
            ##############################
            #  Instrument presence
            if model.double_conditioning is not None:
                instrument_presence_context = \
                    model.data_processor_encodencoder.extract_context_for_generation(frame_index,
                                                                                     context_size,
                                                                                     instruments_presence)
                x_enc_enc, enc_enc_cpc = \
                    model.data_processor_encodencoder.preprocessing(None, None, instrument_presence_context)

                enc_enc_output, *_ = model.encodencoder(x=x_enc_enc,
                                                        cpc=enc_enc_cpc,
                                                        enc_outputs=None,
                                                        enc_enc_outputs=None,
                                                        return_attns=False,
                                                        embed=True,
                                                        mixup_layers=None,
                                                        mixup_lambdas=None)

            ##############################
            # Get piano input
            piano_context = model.data_processor_encoder.extract_context_for_generation(frame_index, context_size,
                                                                                        piano)
            x_enc, cpc_enc = model.data_processor_encoder.preprocessing(piano_context, None, None)

            if model.double_conditioning == 'condition_encoder':
                condition_for_encoder = enc_enc_output
            else:
                condition_for_encoder = None

            enc_output, *_ = model.encoder.forward(
                x=x_enc,
                cpc=cpc_enc,
                enc_outputs=condition_for_encoder,
                enc_enc_outputs=None,
                return_attns=False,
                embed=True,
                mixup_layers=None,
                mixup_lambdas=None
            )

            if model.nade:
                instrument_indices = list(range(model.data_processor_decoder.num_instruments))
                instrument_indices *= number_sampling_steps
                random.shuffle(instrument_indices)
            else:
                instrument_indices = range(model.data_processor_decoder.num_instruments)

            for instrument_index in instrument_indices:

                # Hard switch off orchestra note ?
                hard_switch = False
                if hard_switch:
                    if orchestra_silenced_instruments[instrument_index] == 1:
                        continue

                # Get orchestra input
                orchestra_context = model.data_processor_decoder.extract_context_for_generation(frame_index,
                                                                                                context_size,
                                                                                                orchestra)
                x_dec, cpc_dec = model.data_processor_decoder.preprocessing(None, orchestra_context, None)

                if model.double_conditioning == 'concatenate':
                    condition_for_decoder = torch.cat([enc_enc_output, enc_output], dim=1)
                    condition_for_decoder_2 = None
                elif model.double_conditioning == 'stack_conditioning_layer':
                    condition_for_decoder = enc_output
                    condition_for_decoder_2 = enc_enc_output
                else:
                    condition_for_decoder = enc_output
                    condition_for_decoder_2 = None

                pred_seq, _, _ = model.decoder.forward(
                    x=x_dec,
                    cpc=cpc_dec,
                    enc_outputs=condition_for_decoder,
                    enc_enc_outputs=condition_for_decoder_2,
                    return_attns=False,
                    embed=True,
                    mixup_layers=None,
                    mixup_lambdas=None
                )

                preds = model.data_processor_decoder.pred_seq_to_preds(pred_seq)
                pred = preds[instrument_index]
                # Prediction is in the last frame
                pred_t = pred[:, -1, :]

                prob = F.softmax(pred_t, dim=1)
                p_np = to_numpy(prob)

                #  Zero meta symbols
                start_index = model.data_processor_decoder.dataset.midi_pitch2index[instrument_index][
                    START_SYMBOL]
                end_index = model.data_processor_decoder.dataset.midi_pitch2index[instrument_index][
                    END_SYMBOL]
                pad_index = model.data_processor_decoder.dataset.midi_pitch2index[instrument_index][
                    PAD_SYMBOL]

                p_np[:, start_index] = 0
                p_np[:, end_index] = 0
                p_np[:, pad_index] = 0
                # Normalize
                mean_prob = np.sum(p_np, axis=1, keepdims=True)
                p_np = p_np / mean_prob

                # temperature ?!
                p_temp = np.exp(np.log(p_np) * temperature)
                p = p_temp / np.sum(p_temp, axis=1, keepdims=True)

                for batch_index in range(batch_size):
                    # new_pitch_index = np.argmax(p)
                    predicted_one_hot_value = np.random.choice(np.arange(len(p[0])), p=p[batch_index])
                    orchestra[batch_index, frame_index, instrument_index] = int(predicted_one_hot_value)
    return orchestra


def generation_arrangement_entropy_based_ordering(self,
                                                  piano,
                                                  orchestra_init,
                                                  orchestra_silenced_instruments,
                                                  temperature=1.,
                                                  batch_size=2,
                                                  lowest_entropy_first=True,
                                                  plot_attentions=False,
                                                  events=None
                                                  ):
    self.eval()

    cpc_flag = hasattr(self.data_processor_decoder, 'cpc_model')
    if cpc_flag:
        raise NotImplementedError

    context_size = self.data_processor_decoder.num_frames_orchestra - 1

    with torch.no_grad():

        # Parameters
        num_instruments = self.data_processor_decoder.num_instruments
        first_frame = context_size
        last_frame = piano.size()[1] - 1 - context_size

        orchestra = orchestra_init

        # Generate frames one by one
        if events is None:
            events = range(first_frame, last_frame + 1)
        for frame_index in events:
            # Get context
            start_frame = frame_index - context_size
            end_frame = frame_index + context_size
            piano_context = piano[:, start_frame:end_frame + 1, :]

            x_enc = self.data_processor_encoder.preprocessing(piano_context, None, None)

            return_encoder = self.encoder.forward(
                x=x_enc,
                cpc=cpc_enc,
                enc_outputs=None,
                return_attns=plot_attentions,
                embed=True,
            )

            if plot_attentions:
                enc_outputs, enc_slf_attn, _ = return_encoder
            else:
                enc_outputs, = return_encoder

            instruments_still_unknown = list(range(num_instruments))
            for instrument_index, is_silence in enumerate(orchestra_silenced_instruments):
                if is_silence == 1:
                    instruments_still_unknown.remove(instrument_index)

            instruments_still_unknown_batch = [instruments_still_unknown[:] for _ in range(batch_size)]

            while len(instruments_still_unknown_batch[0]) > 0:
                orchestra_context = orchestra[:, start_frame:end_frame + 1, :]
                x_dec = self.data_processor_decoder.preprocessing(None, orchestra_context, None)

                return_decoder = self.decoder.forward(
                    x=x_dec,
                    cpc=cpc_enc,
                    enc_outputs=enc_outputs,
                    embed=True,
                    return_attns=plot_attentions
                )

                if plot_attentions:
                    pred_seq, dec_slf_attn, dec_enc_attn = return_decoder
                else:
                    pred_seq, = return_decoder

                preds = self.data_processor_decoder.pred_seq_to_preds(pred_seq)
                probs = []
                for pred in preds:
                    pred_t = pred[:, -1, :]
                    prob_t = F.softmax(pred_t, dim=1)
                    probs.append(to_numpy(prob_t))

                # find instrument_index looking at entropy.
                # Might be different for each batch. Fuck that we just loop
                for batch_index in range(batch_size):

                    #  Assign zero probability to meta symbols and compute entropies
                    entropies = []

                    for instrument_index in range(num_instruments):
                        #  Zero
                        start_index = self.data_processor_decoder.dataset.midi_pitch2index[instrument_index][
                            START_SYMBOL]
                        end_index = self.data_processor_decoder.dataset.midi_pitch2index[instrument_index][
                            END_SYMBOL]
                        pad_index = self.data_processor_decoder.dataset.midi_pitch2index[instrument_index][
                            PAD_SYMBOL]
                        probs[instrument_index][batch_index, start_index] = 0
                        probs[instrument_index][batch_index, end_index] = 0
                        probs[instrument_index][batch_index, pad_index] = 0

                        # Normalize
                        mean_prob = np.sum(probs[instrument_index][batch_index])
                        probs[instrument_index][batch_index] = probs[instrument_index][batch_index] / mean_prob

                        #  Entropy
                        this_p = probs[instrument_index][batch_index]
                        entropies.append(-(this_p * np.log(this_p + 1e-10)).sum())

                    ##############################
                    # Choose instrument with highest entropy
                    if lowest_entropy_first:
                        next_indices = np.argsort(entropies)
                    else:
                        next_indices = np.argsort(entropies)[::-1]

                    ##############################
                    for k, next_index in enumerate(next_indices):
                        if next_index in instruments_still_unknown_batch[batch_index]:
                            instrument_index = next_index
                            break

                    #  Remove it from the list of instruments still to be sampled
                    instruments_still_unknown_batch[batch_index].remove(instrument_index)

                    # temperature ?!
                    p_temp = np.exp(np.log(probs[instrument_index][batch_index]) * temperature)
                    p = p_temp / np.sum(p_temp)

                    predicted_one_hot_value = np.random.choice(np.arange(len(p)), p=p)
                    orchestra[batch_index, frame_index, instrument_index] = int(predicted_one_hot_value)
    return orchestra


def generation_reduction(self, piano_init, orchestra, temperature=1.0, batch_size=None, plot_attentions=False,
                         events=None):
    self.eval()

    cpc_flag = hasattr(self.data_processor_decoder, 'cpc_model')
    if cpc_flag:
        raise NotImplementedError

    context_size_orchestra = (self.data_processor_decoder.num_frames_orchestra - 1) // 2

    with torch.no_grad():

        # Parameters
        first_frame = context_size_orchestra
        last_frame = orchestra.size()[1] - 1 - context_size_orchestra

        piano = piano_init

        # Generate frames one by one
        if events is None:
            events = range(first_frame, last_frame + 1)

        for frame_index in events:
            # Get context
            start_frame = frame_index - context_size_orchestra
            end_frame = frame_index + context_size_orchestra
            orchestra_context = orchestra[:, start_frame:end_frame + 1, :]

            # Mean attention (over pitch sampling) for plot
            if plot_attentions:
                mean_enc_dec_att = None
                mean_dec_att = None

            x_enc = self.data_processor_encoder.preprocessing(None, orchestra_context, None)

            return_encoder = self.encoder.forward(
                x=x_enc,
                enc_outputs=None,
                return_attns=plot_attentions,
                embed=True,
            )

            if plot_attentions:
                enc_outputs, enc_slf_attn, _ = return_encoder
            else:
                enc_outputs, = return_encoder

            pitch_indices = range(self.data_processor_decoder.num_pitch_piano)

            for pitch_index in pitch_indices:

                piano_context = piano[:, start_frame:end_frame + 1, :]
                x_dec = self.data_processor_decoder.preprocessing(piano_context, None, None)

                return_decoder = self.decoder.forward(
                    x=x_dec,
                    enc_outputs=enc_outputs,
                    embed=True,
                    return_attns=plot_attentions
                )

                if plot_attentions:
                    pred_seq, dec_slf_attn, dec_enc_attn = return_decoder
                    if mean_enc_dec_att is None:
                        mean_enc_dec_att = dec_enc_attn
                        mean_dec_att = dec_slf_attn
                    else:
                        mean_enc_dec_att = [a + b for a, b in zip(mean_enc_dec_att, dec_enc_attn)]
                        mean_dec_att = [a + b for a, b in zip(mean_dec_att, dec_slf_attn)]
                else:
                    pred_seq, = return_decoder

                preds = self.data_processor_decoder.pred_seq_to_preds(pred_seq)

                pred = preds[pitch_index]
                # Prediction is in the last frame
                pred_t = pred[:, -1, :]

                prob = F.softmax(pred_t, dim=1)
                # temperature ?!
                p_temp = torch.exp(torch.log(prob) * temperature)
                p = torch.div(p_temp, torch.sum(p_temp, 1, keepdim=True))

                predicted_one_hot_value = torch.multinomial(input=p, num_samples=1)[:, 0]
                piano[:, frame_index, pitch_index] = predicted_one_hot_value.int()

                # Old sampling, in numpy and batch_ind per batch_ind
                # for batch_index in range(batch_size):
                #     p_np = to_numpy(p)
                #     predicted_one_hot_value = np.random.choice(np.arange(len(p_np[0])), p=p_np[batch_index])
                #     piano[batch_index, frame_index, pitch_index] = int(predicted_one_hot_value)

            # Only plot batch 0 (all batch are from the same example)
            if plot_attentions:
                for layer in range(self.num_layers):
                    #  Take corresponding layer
                    this_ee = enc_slf_attn[layer]
                    this_de = mean_enc_dec_att[layer] / self.data_processor_decoder.num_pitch_piano
                    this_dd = mean_dec_att[layer] / self.data_processor_decoder.num_pitch_piano

                    #  Reshape
                    n_b, l_dec, l_enc = this_de.size()
                    n_heads = n_b // batch_size
                    this_ee = this_ee.view(batch_size, n_heads, l_enc, l_enc)
                    this_de = this_de.view(batch_size, n_heads, l_dec, l_enc)
                    this_dd = this_dd.view(batch_size, n_heads, l_dec, l_dec)
                    for head_ind in range(n_heads):
                        # Plot
                        plt.imshow(this_ee[0, head_ind].detach().cpu().numpy())
                        plt.savefig(f'plots/ee_{layer}_{head_ind}')
                        plt.imshow(this_de[0, head_ind].detach().cpu().numpy())
                        plt.savefig(f'plots/de_{layer}_{head_ind}')
                        plt.imshow(this_dd[0, head_ind].detach().cpu().numpy())
                        plt.savefig(f'plots/dd_{layer}_{head_ind}')

    return piano


def generation_from_file(model,
                         temperature,
                         batch_size,
                         filepath,
                         write_dir,
                         write_name,
                         banned_instruments,
                         unknown_instruments,
                         writing_tempo,
                         subdivision,
                         number_sampling_steps
                         ):
    print(f'# {filepath}')

    context_size = model.data_processor_decoder.num_frames_orchestra - 1

    #  Load input piano score
    if filepath:
        piano, piano_write, rhythm_piano, orchestra_init, \
        instruments_presence, orchestra_silenced_instruments, orchestra_unknown_instruments = \
            model.data_processor_encoder.dataset.init_generation_filepath(batch_size, context_size, filepath,
                                                                          banned_instruments=banned_instruments,
                                                                          unknown_instruments=unknown_instruments,
                                                                          subdivision=subdivision)
    else:
        raise Exception("deprecated")
        # piano, rhythm_piano, orchestra_init, orchestra_silenced_instruments = \
        #     self.data_processor_encoder.init_generation(banned_instruments=banned_instruments,
        #                                                 unknown_instruments=unknown_instruments)

    # if self.nade:
    #     orchestra = self.generation_arrangement_entropy_based_ordering(
    #         piano=piano,
    #         orchestra_init=orchestra_init,
    #         orchestra_silenced_instruments=orchestra_silenced_instruments,
    #         temperature=temperature,
    #         batch_size=batch_size,
    #         lowest_entropy_first=True,
    #         plot_attentions=plot_attentions
    #     )
    # elif self.double_conditioning:
    #     # TODO Modify orchestra_silenced_instruments with UNKNOWN instruments
    #     #  Use YES, NO and UNKNOWN to be sure mapping is correcly done
    #     orchestra = self.generation_arrangement_instruments_presence(
    #         piano=piano,
    #         orchestra_init=orchestra_init,
    #         orchestra_silenced_instruments=orchestra_silenced_instruments,
    #         orchestra_unknown_instruments=orchestra_unknown_instruments,
    #         temperature=temperature,
    #         batch_size=batch_size,
    #         plot_attentions=plot_attentions
    #     )
    # else:
    orchestra = generation_arrangement(
        model=model,
        piano=piano,
        orchestra_init=orchestra_init,
        orchestra_silenced_instruments=orchestra_silenced_instruments,
        instruments_presence=instruments_presence,
        temperature=temperature,
        batch_size=batch_size,
        number_sampling_steps=number_sampling_steps
    )

    piano_cpu = piano_write[:, context_size:-context_size]
    orchestra_cpu = orchestra[:, context_size:-context_size].cpu()
    # Last duration will be a quarter length
    duration_piano = np.asarray(list(rhythm_piano[1:]) + [subdivision]) - np.asarray(list(rhythm_piano[:-1]) + [0])

    for batch_index in range(batch_size):
        model.dataset.visualise_batch(piano_cpu[batch_index], orchestra_cpu[batch_index], duration_piano,
                                      writing_dir=write_dir, filepath=f"{write_name}_{batch_index}",
                                      writing_tempo=writing_tempo, subdivision=subdivision)
    return


def reduction_from_file(self,
                        temperature=1.,
                        batch_size=2,
                        filepath=None,
                        write_name=None,
                        plot_attentions=False,
                        overfit_flag=False,
                        writing_tempo='adagio',
                        subdivision=None
                        ):
    print(f'# {filepath}')

    context_size = self.data_processor_decoder.num_frames_piano - 1

    #  Load input piano score
    piano_init, rhythm_orchestra, orchestra = self.data_processor_encoder.init_reduction_filepath(batch_size,
                                                                                                  filepath,
                                                                                                  subdivision=subdivision)
    ttt = time.time()
    piano = self.generation_reduction(
        piano_init=piano_init,
        orchestra=orchestra,
        temperature=temperature,
        batch_size=batch_size,
        plot_attentions=plot_attentions
    )
    ttt = time.time() - ttt
    print(f"T: {ttt}")

    piano_cpu = piano[:, context_size:-context_size].cpu()
    orchestra_cpu = orchestra[:, context_size:-context_size].cpu()
    # Last duration will be a quarter length
    duration_piano = np.asarray(list(rhythm_orchestra[1:]) + [subdivision]) - np.asarray(
        list(rhythm_orchestra[:-1]) + [0])
    if overfit_flag:
        writing_dir = self.log_dir_overfitted
    else:
        writing_dir = self.log_dir

    for batch_index in range(batch_size):
        self.dataset.visualise_batch(piano_cpu[batch_index], orchestra_cpu[batch_index], duration_piano,
                                     writing_dir=writing_dir, filepath=f"{write_name}_{batch_index}",
                                     writing_tempo=writing_tempo, subdivision=subdivision)
    return


def generation_bach(model, temperature, ascii_melody=None, batch_size=1, force_melody=False,
                    monophonic_conditioning=False):
    model.eval()
    cpc_flag = hasattr(model.data_processor_decoder, 'cpc_model')
    subdivision = model.dataset.subdivision
    if cpc_flag:
        num_tokens_per_block = model.data_processor_decoder.cpc_model.dataloader_generator.num_tokens_per_block
        num_beats_per_block = num_tokens_per_block // (model.data_processor_decoder.num_voices * subdivision)

    num_measures = len(ascii_melody) // 4 // 4
    if cpc_flag:
        sequences_size = int(model.dataset.sequences_size - num_beats_per_block)
    else:
        sequences_size = int(model.dataset.sequences_size)
    constraints_size = int(sequences_size // 2)
    context_size = constraints_size

    with torch.no_grad():
        chorale, constraint_chorale = \
            model.data_processor_decoder.init_generation(num_measures,
                                                         ascii_melody=ascii_melody,
                                                         append_beginning=context_size * subdivision,
                                                         append_end=constraints_size * subdivision)

        # Duplicate along batch dimension
        chorale = chorale.repeat(batch_size, 1, 1)
        constraint_chorale = constraint_chorale.repeat(batch_size, 1, 1)
        if monophonic_conditioning:
            constraint_chorale = constraint_chorale[:, 0:1, :]

        if force_melody:
            chorale[:, 0, :] = constraint_chorale[:, 0, :]

        exclude_symbols = ['START', 'END']

        for beat_index in range(4 * num_measures):
            # iterations per beat
            # mandatory -> position

            for tick_index in range(subdivision):
                remaining_tick_index = 3 - tick_index

                for voice_index in range(4):

                    if (voice_index == 0) and force_melody:
                        continue

                    if cpc_flag:
                        if beat_index >= num_beats_per_block:
                            #  Need an extra block for CPC since sequences are shifted
                            chunk_encoder = \
                                constraint_chorale[:, :, (beat_index - num_beats_per_block) * subdivision:
                                                         (beat_index + sequences_size) * subdivision]
                        else:
                            # Add Pads if we are too early in the sequence
                            pad_notes = torch.Tensor([[
                                [model.dataset.note2index_dicts[0][PAD_SYMBOL]],
                                [model.dataset.note2index_dicts[1][PAD_SYMBOL]],
                                [model.dataset.note2index_dicts[2][PAD_SYMBOL]],
                                [model.dataset.note2index_dicts[3][PAD_SYMBOL]]
                            ]]).long()
                            pad_chunk = pad_notes.repeat(batch_size, 1, num_beats_per_block * subdivision).cuda()
                            chunk_encoder = constraint_chorale[:, :,
                                            beat_index * subdivision:(beat_index + sequences_size) * subdivision]
                            chunk_encoder = torch.cat([pad_chunk, chunk_encoder], dim=2)

                    else:
                        chunk_encoder = constraint_chorale[:, :,
                                        beat_index * subdivision:(beat_index + sequences_size) * subdivision]

                    x_enc, cpc_enc = model.data_processor_encoder.preprocessing(chunk_encoder)

                    enc_outputs, *_ = model.encoder.forward(
                        x=x_enc,
                        cpc=cpc_enc,
                        enc_outputs=None,
                        enc_enc_outputs=None,
                        return_attns=False,
                        embed=True,
                        mixup_layers=None,
                        mixup_lambdas=None
                    )

                    if cpc_flag:
                        if beat_index >= num_beats_per_block:
                            #  Need an extra block for CPC since sequences are shifted
                            chunk_decoder = \
                                chorale[:, :, (beat_index - num_beats_per_block) * subdivision:
                                              (beat_index + sequences_size) * subdivision]
                        else:
                            # Add Pads if we are too early in the sequence
                            pad_notes = torch.Tensor([[
                                [model.dataset.note2index_dicts[0][PAD_SYMBOL]],
                                [model.dataset.note2index_dicts[1][PAD_SYMBOL]],
                                [model.dataset.note2index_dicts[2][PAD_SYMBOL]],
                                [model.dataset.note2index_dicts[3][PAD_SYMBOL]]
                            ]]).long()
                            pad_chunk = pad_notes.repeat(batch_size, 1, num_beats_per_block * subdivision).cuda()
                            chunk_decoder = chorale[:, :,
                                            beat_index * subdivision:(beat_index + sequences_size) * subdivision]
                            chunk_decoder = torch.cat([pad_chunk, chunk_decoder], dim=2)

                    else:
                        chunk_decoder = chorale[:, :,
                                        beat_index * subdivision:(beat_index + sequences_size) * subdivision]

                    x_dec, cpc_dec = model.data_processor_decoder.preprocessing(chunk_decoder)
                    pred_seq, *_ = model.decoder.forward(
                        x=x_dec,
                        cpc=cpc_dec,
                        enc_outputs=enc_outputs,
                        enc_enc_outputs=None,
                        return_attns=False,
                        embed=True,
                        mixup_layers=None,
                        mixup_lambdas=None
                    )
                    preds = model.data_processor_decoder.pred_seq_to_preds(pred_seq)

                    probs = F.softmax(
                        preds[voice_index][:, -1 - remaining_tick_index, :],
                        dim=1)

                    p = to_numpy(probs)
                    # temperature ?!
                    p = np.exp(np.log(p + 1e-20) * temperature)

                    # exclude non note symbols:
                    for sym in exclude_symbols:
                        sym_index = model.dataset.note2index_dicts[voice_index][sym]
                        p[:, sym_index] = 0
                    p = p / p.sum(axis=1, keepdims=True)

                    for batch_index in range(batch_size):
                        new_pitch_index = np.random.choice(np.arange(
                            model.data_processor_decoder.num_notes_per_voice[voice_index]
                        ), p=p[batch_index])
                        # new_pitch_index = np.argmax(p)
                        chorale[batch_index, voice_index,
                                beat_index * subdivision + context_size * subdivision - remaining_tick_index - 1] = \
                            int(
                                new_pitch_index)
                        # constraint_chorale[:, voice_index,
                        # beat_index * subdivision + context_size * subdivision -
                        # remaining_tick_index - 1] = int(new_pitch_index)

    scores = []
    for batch_index in range(batch_size):
        scores.append(model.dataset.tensor_to_score(
            chorale[batch_index]
        ))
    return scores


def generation_bach_nade(self, temperature, ascii_melody=None, batch_size=1, force_melody=False):
    self.eval()

    cpc_flag = hasattr(self.data_processor_decoder, 'cpc_model')
    if cpc_flag:
        raise NotImplementedError

    num_measures = len(ascii_melody) // 4 // 4
    assert self.dataset.sequences_size % 4 == 0
    subdivision = self.dataset.subdivision
    sequences_size_ticks = self.dataset.sequences_size * subdivision

    # constraints_size = sequences_size // 2 - 1
    context_size = sequences_size_ticks // 2
    constraint_size = sequences_size_ticks // 4

    raise Exception("Batch not writen")
    raise Exception("force_melody not writen")

    exclude_symbols = ['START', 'END']

    with torch.no_grad():
        chorale, constraint_chorale = self.data_processor_decoder.init_generation(num_measures,
                                                                                  ascii_melody=ascii_melody,
                                                                                  context_size=0,
                                                                                  append_end_size=0)
        # Append context and constraint chunks
        constraint_chorale = torch.cat([chorale[:, :, :context_size],
                                        constraint_chorale,
                                        chorale[:, :, :constraint_size]], dim=2)

        constraint_chorale = constraint_chorale.repeat(batch_size, 1, 1)

        # No encoder here, directly constraint using
        chorale = constraint_chorale

        total_num_ticks = chorale.shape[2]

        # Generate patches (in beats, not ticks)
        patches = []
        patch_hop_size = sequences_size_ticks // 4
        for start_ind in range(0, total_num_ticks - sequences_size_ticks + 1, patch_hop_size):
            end_ind = start_ind + sequences_size_ticks
            patches.append((start_ind, end_ind))

        for patch_ind, (start_ind, end_ind) in enumerate(patches):
            #  Extract subpatch
            start_tick = start_ind
            end_tick = end_ind

            #  Ticks to sample
            indices_to_sample = []
            time_indices_to_sample = range(context_size, sequences_size_ticks - constraint_size)
            instrument_indices_to_sample = range(4)
            for tt in time_indices_to_sample:
                for ii in instrument_indices_to_sample:
                    indices_to_sample.append((tt, ii))

            while len(indices_to_sample) > 0:
                # Important to extract subpatch here and not before because chorale is updated at each iteration
                subpatch = chorale[:, :, start_tick:end_tick]

                pred_seq, *_ = self.decoder.forward(
                    x=subpatch,
                    enc_outputs=None
                )

                preds = self.data_processor_decoder.pred_seq_to_preds(pred_seq)

                lowest_entropy = float('infinity')
                prob_to_sample = None
                voice_chosen = None
                time_chosen = None
                for time_to_sample, voice_to_sample in indices_to_sample:
                    prob = to_numpy(F.softmax(preds[voice_to_sample][0, time_to_sample], dim=0))

                    # exclude non note symbols:
                    for sym in exclude_symbols:
                        sym_index = self.dataset.note2index_dicts[voice_to_sample][sym]
                        prob[sym_index] = 0
                    prob = prob / sum(prob)

                    entropy = -(prob * np.log(prob + 1e-20)).sum()
                    if entropy < lowest_entropy:
                        lowest_entropy = entropy
                        prob_to_sample = prob
                        voice_chosen = voice_to_sample
                        time_chosen = time_to_sample

                # Temperature
                p = np.exp(np.log(prob_to_sample + 1e-20) * temperature)
                p = p / sum(p)

                #  Sample
                new_pitch_index = np.random.choice(np.arange(
                    self.data_processor_decoder.num_notes_per_voice[voice_chosen]
                ), p=p)

                #  Write in chorale
                chorale[:, voice_chosen, start_tick + time_chosen] = int(new_pitch_index)

                # Remove from the list of indices to sample
                indices_to_sample.remove((time_chosen, voice_chosen))

    tensor_score = chorale
    score = self.dataset.tensor_to_score(
        tensor_score[0]
    )
    return score, tensor_score, None
