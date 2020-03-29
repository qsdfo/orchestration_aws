import math
import os
from itertools import islice

import matplotlib
import torch
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm

from Transformer.attention_module import AttentionModule
from Transformer.helpers import dict_pretty_print
from Transformer.loss import mean_crossentropy, mean_crossentropy_mixup

matplotlib.use('Agg')


class Transformer(nn.Module):
    def __init__(self,
                 dataset,
                 data_processor_encodencoder,
                 data_processor_encoder,
                 data_processor_decoder,
                 num_heads,
                 per_head_dim,
                 position_ff_dim,
                 enc_dec_conditioning,
                 num_layers,
                 hierarchical_encoding,
                 block_attention,
                 nade,
                 conditioning,
                 double_conditioning,
                 dropout,
                 input_dropout,
                 input_dropout_token,
                 lr,
                 reduction_flag=False,
                 gpu_ids=[0],
                 suffix="",
                 mixup=False,
                 scheduled_training=False
                 ):
        super(Transformer, self).__init__()
        self.input_dropout = input_dropout
        self.input_dropout_token = input_dropout_token
        self.dataset = dataset
        self.conditioning = conditioning
        self.double_conditioning = double_conditioning
        self.nade = nade
        self.reduction_flag = reduction_flag

        # Prevent non sensical model
        assert not self.double_conditioning or self.conditioning

        # Hierarchical means that each layer in the encoder condition the corresponding layer in the decoder,
        # instead of just the final layer of the encoder
        self.hierarchical_encoding = hierarchical_encoding
        self.block_attention = block_attention

        self.data_processor_decoder = data_processor_decoder
        self.data_processor_encoder = data_processor_encoder
        self.data_processor_encodencoder = data_processor_encodencoder

        #  Dimensions
        self.d_model = num_heads * per_head_dim
        self.num_layers = num_layers
        self.suffix = suffix

        #  Mixup
        self.mixup = mixup
        # self.mixup_layers_subset = list(range(self.num_layers))

        # Attention modules
        if self.double_conditioning == 'stack_conditioning_layer':
            double_conditioning = True
        else:
            double_conditioning = False
        self.decoder = nn.DataParallel(AttentionModule(
            n_layers=self.num_layers,
            n_head=num_heads,
            d_k=per_head_dim,
            d_v=per_head_dim,
            d_model=self.d_model,
            d_inner=position_ff_dim,
            data_processor=self.data_processor_decoder,
            enc_dec_conditioning=enc_dec_conditioning,
            dropout=dropout,
            input_dropout=input_dropout,
            double_conditioning=double_conditioning,
            conditioning=self.conditioning,  # must be set to true when it is conditioned
            conditioner=False,
            shift=(not self.nade)
        ), device_ids=gpu_ids, dim=0)

        if self.conditioning:
            if self.double_conditioning in ['condition_encoder']:
                conditioning = True
            else:
                conditioning = False
            self.encoder = nn.DataParallel(AttentionModule(
                n_layers=self.num_layers,
                n_head=num_heads,
                d_k=per_head_dim,
                d_v=per_head_dim,
                d_model=self.d_model,
                d_inner=position_ff_dim,
                data_processor=self.data_processor_encoder,
                enc_dec_conditioning=enc_dec_conditioning,
                dropout=dropout,
                input_dropout=input_dropout,
                double_conditioning=False,
                conditioning=conditioning,
                conditioner=True,
                shift=False
            ), device_ids=gpu_ids, dim=0)

            if self.double_conditioning is not None:
                #  TODO: 1 layer only ?
                #  todo: input dropout also here ? Perhaps on embedding, not on tokens
                #   En gros faut simplifier le modèle
                self.encodencoder = nn.DataParallel(AttentionModule(
                    n_layers=num_layers,
                    n_head=num_heads,
                    d_k=per_head_dim,
                    d_v=per_head_dim,
                    d_model=self.d_model,
                    d_inner=position_ff_dim,
                    data_processor=self.data_processor_encodencoder,
                    enc_dec_conditioning=enc_dec_conditioning,
                    dropout=dropout,
                    input_dropout=input_dropout,
                    double_conditioning=False,
                    conditioning=False,
                    conditioner=True,
                    shift=False
                ), device_ids=gpu_ids, dim=0)

        self.scheduled_training = (scheduled_training > 0)
        self.warmup_pass = scheduled_training
        # If LambdaLR scheduler is used, keep in mind that lr_schedule is multiplied to the learning_rate
        self.lr = lr
        if self.scheduled_training:
            self.optimizer = None
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return

    def __repr__(self):
        if self.reduction_flag:
            name = 'Reducter'
        else:
            name = 'Transformer'
        if self.nade:
            name += '-NADE'
        if self.block_attention:
            name += '-block'
        if self.double_conditioning is not None:
            name += f'-{self.double_conditioning}'
        if self.hierarchical_encoding:
            name += '-hierarchical'
        # if self.mixup:
        #     name += '-mixup'
        ret = f'{name}-{self.num_layers}_{self.dataset.__repr__()}'
        if self.suffix != "":
            ret += f'_{self.suffix}'
        return ret

    @property
    def model_dir(self):
        return f'models/{self.__repr__()}'

    @property
    def log_dir(self):
        log_dir = f'logs/{self.__repr__()}'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    @property
    def log_dir_overfitted(self):
        log_dir_overfitted = f'logs/{self.__repr__()}_OVERFIT'
        if not os.path.exists(log_dir_overfitted):
            os.makedirs(log_dir_overfitted)
        return log_dir_overfitted

    def save(self, score_train=None, score_valid=None, epoch=0):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.state_dict(), f'{self.model_dir}/state_dict')

        if score_train is not None:
            with open(f'{self.model_dir}/score_train.txt', 'w') as ff:
                ff.write(f'{epoch}\n{score_train}')
        if score_valid is not None:
            with open(f'{self.model_dir}/score_valid.txt', 'w') as ff:
                ff.write(f'{epoch}\n{score_valid}')

        print(f'Model {self.__repr__()} saved')

        return

    def save_overfit(self, score_train=None, score_valid=None, epoch=0):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.state_dict(), f'{self.model_dir}/state_dict_overfit')

        if score_train is not None:
            with open(f'{self.model_dir}/score_train_overfit.txt', 'w') as ff:
                ff.write(f'{epoch}\n{score_train}')
        if score_valid is not None:
            with open(f'{self.model_dir}/score_valid_overfit.txt', 'w') as ff:
                ff.write(f'{epoch}\n{score_valid}')

        print(f'Overfitted model {self.__repr__()} saved')
        return

    def load(self, path=None):
        print(f'Loading model {self.__repr__()}')
        if path is None:
            self.load_state_dict(torch.load(f'{self.model_dir}/state_dict'))
        else:
            self.load_state_dict(torch.load(f'{path}/state_dict'))

    def load_overfit(self, path=None, device='cpu'):
        print(f'Loading overfitted model {self.__repr__()}')
        if path is None:
            self.load_state_dict(torch.load(f'{self.model_dir}/state_dict_overfit', map_location=torch.device(device)))
        else:
            self.load_state_dict(torch.load(f'{path}/state_dict_overfit'), map_location=torch.device(device))

    def init_optimizer_and_scheduler(self, generator):
        num_batches = len(generator)
        warmup_steps = num_batches * self.warmup_pass

        lr_max = self.lr / ((self.d_model * warmup_steps) ** (-0.5))
        self.optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-09, lr=lr_max)

        # ------------------------------------------------------------------------
        #  Scheduler
        #  Warmup (cloche avec pic à warmup_steps)
        if self.scheduled_training:
            lr_schedule = lambda epoch: self.d_model ** (-0.5) * min((epoch + 1) ** (-0.5),
                                                                     (epoch + 1) * (warmup_steps ** (-1.5)))
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                               lr_lambda=lr_schedule)

        # Cosine with restart
        # self.T_max = 10
        # self.T_mult = 2
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.T_max)

        #  Reduce LR on plateau
        # torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
        #                                            factor=0.1,
        #                                            patience=10,
        #                                            threshold=0.1,
        #                                            eps=1e-08)
        # ------------------------------------------------------------------------
        return

    def forward(self, x_enc_enc, x_enc, x_dec,
                enc_enc_cpc, enc_cpc, dec_cpc,
                epoch_id, sample_id, label_smoothing,
                loss_type, loss_on_last_frame, return_attns):

        batch_size = x_dec.size()[0]

        #  Mixup
        # if self.mixup and self.training:
        #     # Sample layer
        #     mixup_layers = torch.tensor(random.choices(self.mixup_layers_subset, k=batch_size//2)).float().cuda()
        #     mixup_lambdas = torch.tensor(np.random.uniform(0, 1, batch_size // 2)).float().cuda()
        # else:
        #     mixup_layers = None
        #     mixup_lambdas = None
        mixup_layers = None

        # Mixup
        if self.training and self.mixup:
            lambda_distribution = torch.distributions.beta.Beta(0.2, 0.2)
            mixup_lambdas = lambda_distribution.sample(sample_shape=torch.Size([batch_size // 2])).float().cuda()
        else:
            mixup_lambdas = None

        if self.double_conditioning is not None:
            # Masking on enc_enc information
            if self.training:
                # Todo implement
                # x_enc_enc_masked, _ = self.data_processor_encodencoder.mask_instrument_activations(x_enc_enc)
                x_enc_enc_masked = x_enc_enc
            else:
                x_enc_enc_masked = x_enc_enc
            enc_enc_output, *_ = self.encodencoder(x=x_enc_enc_masked,
                                                   cpc=enc_enc_cpc,
                                                   enc_outputs=None,
                                                   enc_enc_outputs=None,
                                                   return_attns=False,
                                                   embed=True,
                                                   mixup_layers=mixup_layers,
                                                   mixup_lambdas=mixup_lambdas)
        else:
            enc_enc_output = None

        if self.conditioning:
            # For Bach we need masking input,
            # For orchestra, this is identity function (quoique ca pourrait etre utile de masker le piano ??)
            if self.training:
                x_enc_masked, mask_encoder = self.data_processor_encoder.mask_encoder(x_enc, p=self.input_dropout_token)
            else:
                x_enc_masked, mask_encoder = x_enc, None

            # No masking on encoder's inputs
            if self.double_conditioning == 'condition_encoder':
                condition_for_encoder = enc_enc_output
            else:
                condition_for_encoder = None

            enc_output, enc_slf_attn_list, _ = self.encoder(x=x_enc_masked,
                                                            cpc=enc_cpc,
                                                            enc_outputs=condition_for_encoder,
                                                            enc_enc_outputs=None,
                                                            return_attns=return_attns,
                                                            embed=True,
                                                            mixup_layers=mixup_layers,
                                                            mixup_lambdas=mixup_lambdas
                                                            )
        else:
            enc_output = None

        # Random padding of decoder input when nade
        if self.training:
            if self.nade:
                # epoch_id needed for mask scheduling
                masked_x_dec, mask_decoder = self.data_processor_decoder.mask_nade(x_dec,
                                                                                   epoch_id=epoch_id)
                # Actually mask the loss with padded frames
                mask_loss = mask_decoder
            else:
                masked_x_dec, mask_decoder = self.data_processor_decoder.mask_decoder(x_dec, p=self.input_dropout_token)
                # Here we can schedule masking over the loss, for instance to learn only on last frame
                #  If None, train on everything
                mask_loss = None
        else:
            masked_x_dec, mask_decoder = x_dec, None
            mask_loss = None

        if self.double_conditioning == 'concatenate':
            condition_for_decoder = torch.cat([enc_enc_output, enc_output], dim=1)
            condition_for_decoder_2 = None
        elif self.double_conditioning == 'stack_conditioning_layer':
            condition_for_decoder = enc_output
            condition_for_decoder_2 = enc_enc_output
        else:
            condition_for_decoder = enc_output
            condition_for_decoder_2 = None

        pred_seq, dec_slf_attn_list, dec_enc_attn_list = \
            self.decoder.forward(x=masked_x_dec,
                                 cpc=dec_cpc,
                                 enc_outputs=condition_for_decoder,
                                 enc_enc_outputs=condition_for_decoder_2,
                                 return_attns=return_attns,
                                 embed=True,
                                 mixup_layers=mixup_layers,
                                 mixup_lambdas=mixup_lambdas
                                 )

        preds = self.data_processor_decoder.pred_seq_to_preds(pred_seq)

        # Super hacky,
        # but Bach is (batch, voice, note)
        #  Arrangement is (batch, time, instrument)
        # Where instrument and voice play the same role...
        # So just permute targets before loss for bach
        targets = self.data_processor_decoder.prepare_target_for_loss(x_dec)

        if mask_loss is not None:
            mask_loss = self.data_processor_decoder.prepare_mask_for_loss(mask_loss)

        if loss_type == "l2":
            criterion = torch.nn.MSELoss()
            preds_flat = torch.stack(preds).permute(1, 0, 2).squeeze()
            targets_flat = targets.squeeze().cuda()
            loss = criterion(preds_flat[:, 1:], targets_flat[:, 1:])
        elif loss_type == 'x_ent':
            if mixup_lambdas is not None:
                loss = mean_crossentropy_mixup(preds=preds,
                                               targets=targets,
                                               mask=mask_loss,
                                               shift=(not self.nade),
                                               mixup_lambdas=mixup_lambdas,
                                               loss_on_last_frame=loss_on_last_frame)
            else:
                loss = mean_crossentropy(preds=preds,
                                         targets=targets,
                                         mask=mask_loss,
                                         shift=(not self.nade),
                                         label_smoothing=label_smoothing,
                                         loss_on_last_frame=loss_on_last_frame)

        # Plot AR
        # if not self.training and (sample_id == 0):
        #     plt.plot(targets_flat[0].cpu().numpy())
        #     plt.savefig(f"/home/leo/Recherche/Code/Transformer/plots/ar/x_{epoch_id}.pdf")
        #     plt.clf()
        #     gen = torch.zeros_like(targets_flat, requires_grad=False).unsqueeze(2) + \
        #           torch.randn_like(targets_flat).unsqueeze(2)
        #     bs, length, dim = gen.size()
        #     for t in range(1, length):
        #         pred_seq, *_ = self.decoder.forward(x=gen,
        #                                             enc_outputs=None,
        #                                             return_attns=False,
        #                                             mixup_layers=None,
        #                                             mixup_lambdas=None
        #                                             )
        #         preds = self.data_processor_decoder.pred_seq_to_preds(pred_seq)
        #         gen[:, t, :] = preds[t]
        #     generation = gen[0].detach().cpu().numpy()
        #     plt.plot(generation)
        #     plt.savefig(f"/home/leo/Recherche/Code/Transformer/plots/ar/y_{epoch_id}.pdf")
        #     plt.clf()
        #
        #     ar = ARMA(generation, order=(1, 0))
        #     model_fit = ar.fit(trend='nc')
        #     print(f"#### PARAMETER: {model_fit.params}")

        if return_attns:
            num_instru = len(preds)
            samples = torch.zeros((batch_size, num_instru))
            for instru_index, pred_instru in enumerate(preds):
                pred_instru_t = pred_instru[:, -1]
                samples[:, instru_index] = torch.argmax(pred_instru_t, -1)
            attentions = {'encoder_self': [e.detach().cpu() for e in enc_slf_attn_list],
                          'decoder_self': [e.detach().cpu() for e in dec_slf_attn_list],
                          'encoder_decoder': [e.detach().cpu() for e in dec_enc_attn_list]}
        else:
            samples = None
            attentions = None

        return {'loss': loss,
                'monitored_quantities': {'loss': loss.mean().item()},
                'samples': samples,
                'attentions': attentions
                }

    def train_model(self,
                    cache_dir,
                    batch_size,
                    num_batches,
                    num_epochs,
                    label_smoothing,
                    loss_on_last_frame,
                    ):

        (generator_train,
         generator_val,
         generator_test) = self.dataset.data_loaders(batch_size=batch_size,
                                                     cache_dir=cache_dir)

        if self.scheduled_training:
            self.init_optimizer_and_scheduler(generator_train)

        best_validation = math.inf

        writer = SummaryWriter(f"runs/{self.__repr__()}")

        for epoch_id in range(num_epochs):

            monitored_quantities_train = self.epoch(
                data_loader=generator_train,
                epoch_id=epoch_id,
                train=True,
                num_batches=num_batches,
                label_smoothing=label_smoothing,
                loss_on_last_frame=loss_on_last_frame,
                return_attns=False
            )

            monitored_quantities_val = self.epoch(
                data_loader=generator_val,
                epoch_id=epoch_id,
                train=False,
                num_batches=num_batches // 4 if num_batches is not None else None,
                label_smoothing=0,
                loss_on_last_frame=loss_on_last_frame,
                return_attns=False
            )

            print(f'======= Epoch {epoch_id} =======')
            print(f'---Train---')
            dict_pretty_print(monitored_quantities_train, endstr=' ' * 5)
            print()
            print(f'---Val---')
            dict_pretty_print(monitored_quantities_val, endstr=' ' * 5)
            print('\n')

            writer.add_scalar('loss_train', monitored_quantities_train["loss"], epoch_id)
            writer.add_scalar('loss_val', monitored_quantities_val["loss"], epoch_id)

            if monitored_quantities_val["loss"] < best_validation:
                print(f'Saving model!')
                self.save(score_train=monitored_quantities_train["loss"],
                          score_valid=monitored_quantities_val["loss"],
                          epoch=epoch_id)
                best_validation = monitored_quantities_val["loss"]
            # Also save overfitted
            self.save_overfit(score_train=monitored_quantities_train["loss"],
                              score_valid=monitored_quantities_val["loss"],
                              epoch=epoch_id)

        writer.close()

    def epoch(self, data_loader, epoch_id, train,
              num_batches,
              label_smoothing,
              loss_on_last_frame,
              return_attns
              ):
        try:
            len_data_loader = len(data_loader)
        except:
            len_data_loader = num_batches
        if num_batches is None or num_batches > len_data_loader:
            num_batches = len_data_loader

        means = None

        if train:
            print(f'lr: {self.optimizer.param_groups[0]["lr"]}')
            self.train()
        else:
            self.eval()

        for sample_id, tensors in tqdm(enumerate(islice(data_loader,
                                                        num_batches))):

            #  Just for checking batch mixing
            # if sample_id == 0:
            #     print(f'{tensors[0].detach().cpu().numpy().max()} - {tensors[0].detach().cpu().numpy().mean()} - '
            #           f'{tensors[1].detach().cpu().numpy().max()} - {tensors[1].detach().cpu().numpy().mean()}')

            if self.double_conditioning:
                x_enc_enc, enc_enc_cpc = self.data_processor_encodencoder.preprocessing(*tensors)
            else:
                x_enc_enc = None
                enc_enc_cpc = None

            if self.conditioning:
                x_enc, enc_cpc = self.data_processor_encoder.preprocessing(*tensors)
            else:
                x_enc = None
                enc_cpc = None

            x_dec, dec_cpc = self.data_processor_decoder.preprocessing(*tensors)

            # SWA
            # self.base_opt.zero_grad()
            self.optimizer.zero_grad()

            forward_pass_gen = self.forward(x_enc_enc, x_enc, x_dec,
                                            enc_enc_cpc, enc_cpc, dec_cpc,
                                            epoch_id, sample_id,
                                            label_smoothing=label_smoothing,
                                            loss_type='x_ent',
                                            loss_on_last_frame=loss_on_last_frame,
                                            return_attns=return_attns)
            loss = forward_pass_gen['loss']

            if train:
                loss.backward()
                # # Todo clip grad?!
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
                # Check gradients
                # params = list(self.named_parameters())
                # mean = 0
                # counter = 0
                # maxi = -float('infinity')
                # mini = float('infinity')
                # for name_param, data_param in params:
                #     # print(f'{name_param} : {data_param.float().mean()}')
                #     param_grad = data_param.grad
                #     if param_grad is not None:
                #         counter += len(param_grad.view(-1))
                #         mean += param_grad.float().sum()
                #         maxi = max(maxi, param_grad.float().max())
                #         mini = min(mini, param_grad.float().min())
                #         mean = mean / counter
                # print(f'Mean:{mean} - Min:{mini} - Max:{maxi}')

                self.optimizer.step()

            #  Update learning rate
            if train and self.scheduled_training:
                self.scheduler.step()

            # Monitored quantities
            monitored_quantities = dict(forward_pass_gen['monitored_quantities'])
            # average quantities
            if means is None:
                means = {key: 0
                         for key in monitored_quantities}
            means = {
                key: value + means[key]
                for key, value in monitored_quantities.items()
            }

            if return_attns:
                attentions = forward_pass_gen['attentions']
                piano_input = x_enc.detach().cpu().numpy()
                orchestra_input = x_dec.detach().cpu().numpy()
                orchestra_output = forward_pass_gen['samples'].unsqueeze(1).detach().numpy()

            del forward_pass_gen
            del loss

        # Re-normalize monitored quantities
        means = {
            key: value / num_batches
            for key, value in means.items()
        }

        if return_attns:
            means['attentions'] = attentions
            means['piano_input'] = piano_input
            means['orchestra_input'] = orchestra_input
            means['orchestra_output'] = orchestra_output

        return means
