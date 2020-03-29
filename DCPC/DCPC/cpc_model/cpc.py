import os
import random
import shutil
from itertools import islice

import numpy as np
import torch
from DCPC.cpc_model.cpc_encoder import CPC_encoder
from DCPC.data_helpers import dict_pretty_print
from DCPC.data_processors.cpc_data_processor import CPCDataProcessor
from DCPC.dataloaders.cpc_dataloader import CPCDataloaderGenerator
from DCPC.loss_functions import nce_loss
from DatasetManager.helpers import REST_SYMBOL
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm


class CPC(nn.Module):
    def __init__(self,
                 dataloader_generator,
                 dataprocessor,
                 vector_quantizer,
                 embedding_size,
                 #
                 num_layers_enc,
                 encoding_size_zt,
                 bidirectional_enc,
                 #
                 num_layers_ar,
                 encoding_size_ct,
                 bidirectional_ar,
                 rnn_hidden_size,
                 #
                 dropout,
                 corrput_labels,
                 lr,
                 beta,
                 model_path,
                 ):
        super(CPC, self).__init__()
        self.dataloader_generator: CPCDataloaderGenerator = dataloader_generator
        self.dataprocessor: CPCDataProcessor = dataprocessor

        self.num_blocks_left = dataloader_generator.num_blocks_left

        self.encoding_size_zt = encoding_size_zt
        self.encoding_size_ct = encoding_size_ct

        self.encoder = CPC_encoder(
            embedding_size,
            #
            num_layers_enc,
            encoding_size_zt,
            bidirectional_enc,
            #
            rnn_hidden_size,
            dropout
        )

        # AR
        self.g_ar_fwd = torch.nn.GRU(
            input_size=encoding_size_zt,
            hidden_size=rnn_hidden_size,
            num_layers=num_layers_ar,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        if bidirectional_ar:
            self.g_ar_bwd = torch.nn.GRU(
                input_size=encoding_size_zt,
                hidden_size=rnn_hidden_size,
                num_layers=num_layers_ar,
                bias=True,
                batch_first=True,
                dropout=dropout,
                bidirectional=False,
            )

        # VQ
        self.vector_quantizer = vector_quantizer
        linear_input_dim = rnn_hidden_size * 2 if bidirectional_ar else rnn_hidden_size
        self.post_rnn_linear_c = torch.nn.Linear(linear_input_dim, encoding_size_ct, bias=True)

        # Classifier on top of CPC encoding used for training only
        self.k_max = self.dataloader_generator.num_blocks_right
        # FIXME which init?
        # self.W = nn.Parameter(torch.randn(encoding_size_zt, encoding_size_ct, self.k_max))
        self.W = nn.Parameter(torch.zeros(encoding_size_zt, encoding_size_ct, self.k_max))

        self.model_path = model_path

        self.beta = beta
        self.corrupt_labels = corrput_labels
        # self.batch_norm = nn.BatchNorm1d(encoding_size_zt, affine=False,
        #                                  track_running_stats=False)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

    # def batch_normalize(self, z_left, z_right, z_negative):
    #     z_left_size = z_left.size()
    #     z_right_size = z_right.size()
    #     z_negative_size = z_negative.size()
    #
    #     z_dim = z_left_size[-1]
    #
    #     flat_input = torch.cat(
    #         (
    #             z_left.view(-1, z_dim),
    #             z_right.view(-1, z_dim),
    #             z_negative.view(-1, z_dim)
    #         ), dim=0
    #     ).unsqueeze(1)
    #
    #     flat_input = flat_input.permute(0, 2, 1)
    #     flat_input = self.batch_norm(flat_input)
    #     flat_input = flat_input.permute(0, 2, 1).contiguous()
    #     flat_input = flat_input[:, 0, :]
    #
    #     # recover original sizes
    #     z_left_batch_size = np.prod(z_left_size[:-1])
    #     z_right_batch_size = np.prod(z_right_size[:-1])
    #     z_negative_batch_size = np.prod(z_negative_size[:-1])
    #
    #     z_left = flat_input[: z_left_batch_size, :].view(z_left_size)
    #     z_right = flat_input[
    #               z_left_batch_size:z_left_batch_size + z_right_batch_size,
    #               :].view(z_right_size)
    #
    #     z_negative = flat_input[
    #                  -z_negative_batch_size:,
    #                  :].view(z_negative_size)
    #     return z_left, z_right, z_negative

    def batch_normalize(self, z_left, z_right, z_negative):
        z_left_size = z_left.size()
        z_right_size = z_right.size()
        z_negative_size = z_negative.size()

        # z_left and z_right
        z_left = self.batch_norm(z_left.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        z_right = self.batch_norm(z_right.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        # z_negative
        z_negative = z_negative.view(
            z_negative.size(0) * z_negative.size(1),
            z_negative.size(2),
            z_negative.size(4)
        )
        z_negative = self.batch_norm(z_negative.permute(0, 2, 1)).permute(0, 2, 1)
        z_negative = z_negative.view(
            *z_negative_size
        ).contiguous()

        return z_left, z_right, z_negative

    def save(self):
        path = self.model_path
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_dict(), f'{path}/state_dict')
        print(f'Model {path} saved')

    def load(self):
        path = self.model_path
        print(f'Loading model {path}')
        self.load_state_dict(torch.load(f'{path}/state_dict'))

    def compute_c(self, zs, h=None):
        c, h = self.g_ar_fwd(zs, h)
        # take last time step
        c = c[:, -1]
        c = self.post_rnn_linear_c(c)
        return c

    def epoch(self,
              data_loader,
              train,
              num_batches
              ):

        means = {
            'loss':                   0,
            'accuracy':               0,
            'loss_quantize':          0,
            'loss_contrastive':       0,
            'num_codewords':          0,
            'num_codewords_negative': 0,
        }

        print(f'lr: {self.optimizer.param_groups[0]["lr"]}')

        if train:
            self.train()
        else:
            self.eval()

        for sample_id, tensor_dict in tqdm(enumerate(islice(data_loader,
                                                            num_batches))):
            # -- to CUDA
            tensor_dict = self.dataprocessor.preprocess(tensor_dict)

            tensor_dict = self.dataprocessor.embed(tensor_dict)

            # todo stack?
            # -- compute z
            z_left = self.encoder.compute_z(tensor_dict['x_left'])
            z_right = self.encoder.compute_z(tensor_dict['x_right'])
            z_negative = self.encoder.compute_z(tensor_dict['negative_samples'])

            # FIXME remove or to add properly
            # MUST be on z_quantized!
            # (z_left,
            #  z_right,
            #  z_negative) = self.batch_normalize(z_left,
            #                                     z_right,
            #                                     z_negative
            #                                     )

            # quantize
            (z_quantized_negative,
             encodings_negative,
             loss_quantization_negative) = self.vector_quantizer(
                z_negative,
                corrupt_labels=self.corrupt_labels)
            # TODO corrupt here also?
            (z_quantized_left,
             encodings_left,
             loss_quantization_left) = self.vector_quantizer(z_left,
                                                             corrupt_labels=False)
            # corrupt_labels=self.corrupt_labels)
            (z_quantized_right,
             encodings_right,
             loss_quantization_right) = self.vector_quantizer(z_right, corrupt_labels=False)

            # -- compute c
            c = self.compute_c(z_quantized_left, h=None)

            #  -- Positive fks
            fks_positive = self.compute_fks(c, z_quantized_right)

            #  --Negative fks

            # z_negative is
            # (batch_size, num_negative_samples, num_blocks_right, 1, z_dim)
            # remove unused dim # fixme
            z_quantized_negative = z_quantized_negative[:, :, :, 0, :]
            (batch_size,
             num_negative_samples,
             num_blocks_right,
             z_dim) = z_quantized_negative.size()

            z_quantized_negative = z_quantized_negative.permute(1, 0, 2, 3).contiguous().view(
                batch_size * num_negative_samples,
                num_blocks_right,
                z_dim
            )

            c_repeat = c.repeat(num_negative_samples, 1)
            fks_negative = self.compute_fks(c_repeat, z_quantized_negative)

            fks_negative = fks_negative.view(num_negative_samples,
                                             batch_size,
                                             num_blocks_right
                                             ).contiguous().permute(1, 2, 0)

            # fks_negative is now (batch_size, k, num_negative_examples)
            # fks_positive is (batch_size, k)

            # -- compute score:

            score_matrix = fks_positive > fks_negative.max(2)[0]
            #########################

            # == Compute loss
            # -- contrastive loss
            contrastive_loss = nce_loss(fks_positive, fks_negative)

            loss_quantization = self._loss_quantization(loss_quantization_left,
                                                        loss_quantization_negative,
                                                        loss_quantization_right)

            loss = contrastive_loss + self.beta * loss_quantization
            # loss = contrastive_loss

            # == Optim
            self.optimizer.zero_grad()
            if train:
                loss.backward()
                # TODO clip grad norm?
                # nn.utils.clip_grad_value_(self.parameters(), clip_value=3)
                self.optimizer.step()
            #########################

            # Monitored quantities and clean
            means['loss'] += loss.item()
            means['loss_quantize'] += loss_quantization.item()
            means['loss_contrastive'] += contrastive_loss.item()

            # compute num codewords
            if encodings_left is not None:
                means['num_codewords'] += len(torch.unique(
                    torch.cat((encodings_left, encodings_right), dim=0)
                ))
                means['num_codewords_negative'] += len(torch.unique(
                    encodings_negative
                ))

            del contrastive_loss
            del loss_quantization
            del loss

            accuracy = score_matrix.sum(dim=0).float() / batch_size
            means['accuracy'] += accuracy.detach().cpu().numpy()
            del accuracy

        if num_batches is None:
            num_batches = sample_id + 1
        # Re-normalize monitored quantities and free gpu memory
        means = {
            key: (value / num_batches)
            for key, value in means.items()
        }

        means['accuracy'] = list(means['accuracy'])

        return means

    def _loss_quantization(self,
                           loss_quantization_left,
                           loss_quantization_negative,
                           loss_quantization_right):
        # TODO other quantization loss?
        # -- quantization loss
        # loss_quantization = torch.cat(
        #     (loss_quantization_left.view(-1),
        #      loss_quantization_right.view(-1),
        #      loss_quantization_negative.view(-1),
        #      ), dim=0
        # ).mean()
        loss_quantization = torch.cat(
            (loss_quantization_left.sum(1),
             loss_quantization_right.sum(1),
             loss_quantization_negative.sum(3).sum(2).sum(1),
             ), dim=0
        ).mean()
        # loss_quantization = (loss_quantization_left.mean()
        #                      + loss_quantization_right.mean()
        #                      + loss_quantization_negative.mean())
        return loss_quantization

    def train_model(self,
                    batch_size,
                    num_epochs,
                    num_batches,
                    num_negative_samples
                    ):
        writer = SummaryWriter(self.model_path)

        for epoch_id in range(num_epochs):
            (generator_train,
             generator_val,
             generator_test) = self.dataloader_generator.dataloader(
                batch_size=batch_size,
                num_negative_samples=num_negative_samples)

            monitored_quantities_train = self.epoch(
                data_loader=generator_train,
                train=True,
                num_batches=num_batches,
            )

            monitored_quantities_val = self.epoch(
                data_loader=generator_val,
                train=False,
                num_batches=num_batches // 4,
            )

            # for debug ===
            # (generator_train,
            #  generator_val,
            #  generator_test) = self.dataloader_generator.dataloader(
            #     batch_size=batch_size,
            #     num_negative_samples=num_negative_samples)
            # monitored_quantities_val = self.epoch(
            #     data_loader=generator_train,
            #     train=True,
            #     num_batches=num_batches // 4,
            # )
            # ====

            # === Logging
            print(f'======= Epoch {epoch_id} =======')
            print(f'---Train---')
            dict_pretty_print(monitored_quantities_train, endstr=' ' * 5)
            print()
            print(f'---Val---')
            dict_pretty_print(monitored_quantities_val, endstr=' ' * 5)
            print('\n')

            # --- Tensorboard
            # - Losses
            writer.add_scalar('loss/train', monitored_quantities_train["loss"], epoch_id)
            writer.add_scalar('loss/val', monitored_quantities_val["loss"], epoch_id)

            # - Contrastive
            writer.add_scalar('loss_contrastive/train', monitored_quantities_train[
                "loss_contrastive"],
                              epoch_id)
            writer.add_scalar('loss_contrastive/val', monitored_quantities_val["loss_contrastive"],
                              epoch_id)

            # - Quantization
            writer.add_scalar('loss_quantize/train', monitored_quantities_train[
                "loss_quantize"],
                              epoch_id)
            writer.add_scalar('loss_quantize/val', monitored_quantities_val["loss_quantize"],
                              epoch_id)

            # - Num codewords
            writer.add_scalar('num_codewords/train', monitored_quantities_train[
                "num_codewords"],
                              epoch_id)
            writer.add_scalar('num_codewords_negative/train', monitored_quantities_train[
                "num_codewords_negative"],
                              epoch_id)

            writer.add_scalar('num_codewords/val', monitored_quantities_val[
                "num_codewords"],
                              epoch_id)
            writer.add_scalar('num_codewords_negative/val', monitored_quantities_val[
                "num_codewords_negative"],
                              epoch_id)

            # - Accuracies
            for k, acc in enumerate(monitored_quantities_train['accuracy']):
                writer.add_scalar(f'Training Accuracy/{k + 1}', acc, epoch_id)
            for k, acc in enumerate(monitored_quantities_val['accuracy']):
                writer.add_scalar(f'Validation Accuracy/{k + 1}', acc, epoch_id)
            print(f'Saving model!')
            self.save()

        writer.close()

    def visualize_encoding_bach(self, config):
        #  Important to call at each epoch since not all elements are used as positive particles
        batch_size = 2048
        generators = self.dataset.data_loaders(batch_size=batch_size)
        num_negative_particles = None

        subdivision = config['subdivision']
        num_samples_per_unit = config['num_samples_per_unit']
        num_voices = 4

        silence_chord = np.array(
            [note2index[REST_SYMBOL] for note2index in self.dataset.note2index_dicts])
        silence_beat = np.array([silence_chord] * subdivision).T

        root_save_folder = f'../logs/{config["savename"]}/visualize_chords'
        if os.path.isdir(root_save_folder):
            shutil.rmtree(root_save_folder)
        os.makedirs(root_save_folder)

        samples_sorted_max_encoding = dict()

        # for generator in list(generators):
        #  Only use train samples ?
        generators = generators[0]

        # for generator in generators:
        for tensors in tqdm(generators):
            xs = self.data_processor.preprocessing(num_negative_particles, *tensors)
            # Embed
            x_embed = self.data_processor.embed(xs)
            # Compute and flatten z_t
            _, zs, _, _, _ = self.encoder(x_embed, h_cs=None)
            zs = zs.detach().cpu().numpy()
            xs = xs.detach().cpu().numpy()
            length = zs.shape[1]
            zs_flat = zs.reshape(batch_size * length, -1)
            for unit_index in range(self.encoding_size_zt):
                #  Get the position in batch of the units which maximally activate the zt
                zs_unit = zs_flat[:, unit_index]
                max_zs = zs_unit.argsort()[-10 * num_samples_per_unit:][::-1]
                ind_max_zs = np.unravel_index(max_zs, zs[:, :, 0].shape)
                max_elem_candidates = []
                for this_example_ind in range(10 * num_samples_per_unit):
                    batch_ind_max = ind_max_zs[0][this_example_ind]
                    zt_ind_max = ind_max_zs[1][this_example_ind]
                    start_time = int(zt_ind_max * (self.num_blocks_left / num_voices))
                    end_time = int((zt_ind_max + 1) * (self.num_blocks_left / num_voices))
                    elem = xs[batch_ind_max, :, start_time:end_time]
                    score = zs[batch_ind_max, zt_ind_max, unit_index]
                    flat_elem = list(elem.flatten())
                    skip = False
                    for _, _, flat_comparison in max_elem_candidates:
                        if flat_elem == flat_comparison:
                            skip = True
                    if skip:
                        continue
                    max_elem_candidates.append((elem, score, flat_elem))
                    if len(max_elem_candidates) > num_samples_per_unit:
                        break

                #  Check if there are larger than some units in the stored samples
                if unit_index not in samples_sorted_max_encoding.keys():
                    samples_sorted_max_encoding[unit_index] = max_elem_candidates
                else:
                    #  Append to existing list, sort, keep num_samples first elems
                    temp_list = samples_sorted_max_encoding[unit_index]
                    for candidate_elem in max_elem_candidates:
                        skip = False
                        for _, _, flat_comparison in temp_list:
                            if flat_comparison == candidate_elem[2]:
                                skip = True
                                break
                        if skip:
                            continue
                        else:
                            temp_list.append(candidate_elem)

                    temp_list = sorted(temp_list, key=lambda x: -x[1])
                    samples_sorted_max_encoding[unit_index] = temp_list[:num_samples_per_unit]

        #  Write scores
        for unit_index, list_chords_and_score in samples_sorted_max_encoding.items():
            save_midi_path = f'{root_save_folder}/{unit_index}.mid'
            #  Create a list with only the chords (not the score), and with silences between each element
            list_chords = []
            for elem in list_chords_and_score:
                list_chords.append(elem[0])
                list_chords.append(silence_beat)
            chords_matrix = np.concatenate(list_chords, axis=1)
            score = self.dataset.tensor_to_score(chords_matrix)
            score.write('mid', save_midi_path)

        return

    def visualize_discrete_encoding_bach(self):
        # TODO make this class general
        self.eval()
        batch_size = 1024

        (dataloader_train,
         dataloader_val,
         dataloader_test) = self.dataloader_generator.block_dataloader(
            batch_size
        )
        # Create data_loader from reference_dataset
        reference_dataset = self.dataprocessor.reference_dataset
        subdivision = reference_dataset.subdivision
        silence_chord = np.array(
            [note2index[REST_SYMBOL] for note2index in reference_dataset.note2index_dicts])
        silence_beat = np.array([silence_chord] * subdivision).T

        root_save_folder = f'{self.model_path}/visualize_chords'
        # TODO keep all results?
        if os.path.isdir(root_save_folder):
            shutil.rmtree(root_save_folder)
        os.makedirs(root_save_folder)

        samples_sorted_max_encoding = dict()

        # for generator in list(generators):
        #  Only use train samples ?
        # for generator in generators:
        # TODO use val
        for tensor_dict in tqdm(dataloader_val):

            # Preprocess and embed
            original_x = tensor_dict['x']
            tensor_dict = self.dataprocessor.preprocess(tensor_dict)
            tensor_dict = self.dataprocessor.embed(tensor_dict)

            z = self.encoder.compute_z(x=tensor_dict['x'])
            # quantize
            (z_quantized,
             encodings,
             loss_quantization) = self.vector_quantizer(
                z,
                corrupt_labels=False)
            encoding_indexes = encodings[:, 0].cpu().numpy()

            d = {}
            original_x = original_x.cpu().numpy()
            for exemple_index, encoding_index in enumerate(encoding_indexes):
                if encoding_index not in d:
                    d[encoding_index] = []

                d[encoding_index].append(original_x[exemple_index])

            samples_sorted_max_encoding = d
            break

        #  Write scores
        for unit_index, list_chords_and_score in samples_sorted_max_encoding.items():
            save_midi_path = f'{root_save_folder}/{unit_index}.mid'
            #  Create a list with only the chords (not the score), and with silences between each element
            list_chords = []
            random.shuffle(list_chords_and_score)
            for elem in list_chords_and_score[:50]:
                list_chords.append(elem)
                list_chords.append(silence_beat)
            chords_matrix = np.concatenate(list_chords, axis=1)
            score = reference_dataset.tensor_to_score(chords_matrix)
            score.write('mid', save_midi_path)
            print(f'File {save_midi_path} saved')

    def visualize_keys_bach(self, config):
        self.eval()
        #  Important to call at each epoch since not all elements are used as positive particles
        batch_size = 256
        generators = self.dataset.data_loaders(batch_size=batch_size)
        num_negative_particles = None

        subdivision = config['subdivision']
        num_samples_per_unit = config['num_samples_per_unit']
        num_voices = 4

        silence_chord = np.array(
            [note2index[REST_SYMBOL] for note2index in self.dataset.note2index_dicts])
        silence_beat = np.array([silence_chord] * subdivision).T

        root_save_folder = f'../logs/{config["savename"]}/visualize_keys'
        if os.path.isdir(root_save_folder):
            shutil.rmtree(root_save_folder)
        os.makedirs(root_save_folder)

        samples_sorted_max_encoding = dict()

        # for generator in list(generators):
        #  Only use train samples ?
        generators = generators[0]

        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('TkAgg')

        num_batches = 30

        # for generator in generators:
        for k, tensors in tqdm(enumerate(generators)):
            xs = self.data_processor.preprocessing(num_negative_particles, *tensors)
            # FIXME hardcoded
            # get keys
            m = tensors[1][:, :, :, 2]
            m = m.transpose(1, 2).contiguous().view(batch_size, -1)

            # for measures
            m = m[:, ::16 * 4]
            # for chords
            # m = m[:, ::16]

            # TODO what to put first?!
            m = m.view(-1).cpu().numpy()
            _, _, length = xs.size()

            # Embed
            x_embed = self.data_processor.embed(xs)
            # Compute and flatten z_t
            _, _, _, encoding_indexes = self.encoder.compute_z(x_embed)
            encoding_indexes = encoding_indexes.cpu().numpy()

            assert encoding_indexes.shape == m.shape

            df = []
            for i in range(m.shape[0]):
                df.append([encoding_indexes[i], m[i]])

            if k > num_batches:
                break

        df = pd.DataFrame(df, columns=['cluster', 'key'])
        g = sns.FacetGrid(df,
                          col='cluster', margin_titles=True)
        bins = [-0.5 + i for i in range(16)]
        g.map(plt.hist,
              "key",
              bins=bins
              )
        plt.show()

        return

    def compute_fks(self, c_t, zs):
        """

        :param c_t:
        :param zs:
        :return: log of fks
        """
        batch_size = c_t.shape[0]
        W_c = torch.matmul(c_t, self.W).permute(1, 2, 0)
        product = torch.matmul(W_c.view(batch_size, self.k_max, 1, -1),
                               zs.view(batch_size, self.k_max, -1, 1))
        fks = torch.squeeze(product)
        return fks
