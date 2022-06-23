import torch
import simple_transformer
import torch.nn as nn
import numpy as np


class KoreNetworks(nn.Module):

    def __init__(self, config):
        super(KoreNetworks, self).__init__()

        self.config = config

        self.encoder = simple_transformer.Transformer(
            num_layers=config['num_encode_layers'],
            hidden_size=config['hidden_size'],
            reduction=None
        )

        self.emb_map = nn.Linear(
            config['num_input_map_channels'], config['hidden_size'])
        self.emb_vec = nn.Linear(
            config['num_input_vec_channels'], config['hidden_size'])

        self.pos_encoding = nn.init.normal_(
            nn.Parameter(
                torch.zeros(1, config['size2'], config['hidden_size'])),
            std=1e-4)

        self.cls_v = nn.Linear(config['hidden_size'], 1)

        self.cls_first_layer = nn.Sequential(
            nn.Linear(config['hidden_size'], 3))
        self.cls_launch_point = nn.Linear(config['hidden_size'], config['size2'] - 1)
        self.cls_convert_point = nn.Linear(config['hidden_size'], config['size2'] - 1)
        pass

    def encode_obs(self, obs_vec):
        map_emb = self.emb_map(obs_vec['map'])
        vec_emb = self.emb_vec(obs_vec['vec'])

        map_emb += self.pos_encoding

        map_cls_emb, map_emb = self.encoder(map_emb)

        cls_emb = map_cls_emb + vec_emb

        return {'map_emb': map_emb, 'cls_emb': cls_emb}
        pass

    def forward_v(self, embs):

        return self.cls_v(embs['cls_emb'])
        pass

    def forward_first_layer(self, shipyard_emb):
        return self.cls_first_layer(shipyard_emb)
        pass

    def forward_launch_point(self, shipyard_emb):
        return self.cls_launch_point(shipyard_emb)
        pass

    def forward_convert_point(self, shipyard_emb):
        return self.cls_convert_point(shipyard_emb)

    def parameter_size(self,):
        size = 0
        for name, param in self.named_parameters():
            size += param.numel()
            pass

        return size

    def set_parameter(self, start, end, values: np.ndarray):
        idx = 0
        vsize = end - start

        for param in self.parameters():
            psize = param.numel()
            pstart = idx
            pend = idx + psize

            if start > pend:
                continue
            elif end < pstart:
                break
            else:
                origin_require_grad = param.requires_grad
                param.requires_grad = False
                copy_start = max(start - pstart, 0)
                copy_end = min(psize, end - pstart)
                v_start = max(pstart - start, 0)
                v_end = min(vsize, pend - start)
                param.data.flatten()[copy_start:copy_end].copy_(
                    torch.from_numpy(
                        values[v_start:v_end]))
                param.requires_grad = origin_require_grad
                pass
            pass
        pass

    def copy_parameter(self, start, end):
        idx = 0
        vsize = end - start
        values = np.zeros(vsize, dtype='float32')

        for param in self.parameters():
            psize = param.numel()
            pstart = idx
            pend = idx + psize

            if start > pend:
                continue
            elif end < pstart:
                break
            else:
                copy_start = max(start - pstart, 0)
                copy_end = min(psize, end - pstart)
                v_start = max(pstart - start, 0)
                v_end = min(vsize, pend - start)
                values[v_start: v_end] = param.data.flatten()[copy_start:copy_end].numpy()
                pass
            pass

        return values
        pass
    pass

