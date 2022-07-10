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

        self.decoder_launch = simple_transformer.Transformer(
            num_layers=config['num_decode_layers'],
            hidden_size=config['hidden_size'],
            reduction=None
        )

        self.decoder_convert = simple_transformer.Transformer(
            num_layers=config['num_decode_layers'],
            hidden_size=config['hidden_size'],
            reduction=None
        )

        self.head_size = config['hidden_size'] // self.encoder.num_attention_heads

        self.emb_map = nn.Linear(
            config['num_input_map_channels'], config['hidden_size'])
        self.emb_vec = nn.Sequential(
            nn.Linear(
                config['num_input_vec_channels'], config['hidden_size']),
            nn.ReLU(),
            nn.Linear(config['hidden_size'], config['hidden_size']))

        self.cls_v = nn.Linear(config['hidden_size'], 1)

        self.cls_first_layer = nn.Sequential(
            nn.Linear(config['hidden_size'], 3))
        self.cls_launch_point = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.cls_launch_point2 = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.cls_launch_number = nn.Linear(config['hidden_size'], 2)
        self.cls_convert_point = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.cls_convert_point2 = nn.Linear(config['hidden_size'], config['hidden_size'])

        
        

        self.init_map_structure(config)
        pass

    def init_map_structure(self, config):
        self.map_structure_x = nn.Embedding(config['map_size'], self.head_size)
        self.map_structure_y = nn.Embedding(config['map_size'], self.head_size)
        pos = torch.arange(config['size2'], dtype=torch.long)
        pos_x = pos % config['map_size']
        pos_x_diff = pos_x[:, None] - pos_x[None]
        pos_x_diff[pos_x_diff>config['map_size']/2] -= config['map_size']
        pos_x_diff[pos_x_diff<-config['map_size']/2] += config['map_size']
        pos_y = torch.div(pos, config['map_size'], rounding_mode='floor')
        pos_y_diff = pos_y[:, None] - pos_y[None]
        pos_y_diff[pos_y_diff>config['map_size']/2] -= config['map_size']
        pos_y_diff[pos_y_diff<-config['map_size']/2] += config['map_size']
        pos_x_diff = pos_x_diff + config['map_size'] // 2
        pos_y_diff = pos_y_diff + config['map_size'] // 2

        self.register_buffer('pos_x_diff', pos_x_diff)
        self.register_buffer('pos_y_diff', pos_y_diff)
        pass

    def encode_obs(self, obs_vec):
        map_emb = self.emb_map(obs_vec['map'])
        vec_emb = self.emb_vec(obs_vec['vec'])

        batch_size = map_emb.shape[0]
        max_shipyard = obs_vec['my_shipyard']['mask'].shape[1]
        batch_index = torch.tile(
            torch.arange(batch_size, device=map_emb.device), [max_shipyard, 1]).T
        
        map_emb[batch_index, obs_vec['my_shipyard']['index']] += vec_emb[:, None, :]

        map_structure = self.map_structure_x(self.pos_x_diff) + self.map_structure_y(self.pos_y_diff)
        map_cls_emb, map_emb = self.encoder(map_emb, structure_matrix=map_structure)
        # map_emb += vec_emb[:, None, :]
        # map_emb = torch.tile(vec_emb, (1, self.config['size2'], 1))

        cls_emb = map_cls_emb + vec_emb
        # cls_emb = vec_emb

        return {
            'map_emb': map_emb, 'cls_emb': cls_emb, 'map_structure': map_structure,
            'batch_index': batch_index}
        pass

    def forward_v(self, embs):

        return self.cls_v(embs['cls_emb'])
        pass

    def forward_first_layer(self, shipyard_emb):
        return self.cls_first_layer(shipyard_emb)
        pass

    def forward_launch_point(self, embs, shipyard_index):
        map_structure = embs['map_structure']
        _, map_emb_decode = self.decoder_launch(embs['map_emb'], structure_matrix=map_structure)
        # map_emb: [B, N, D]

        shipyard_emb = map_emb_decode[embs['batch_index'], shipyard_index]
        # shipyard_emb: [B, SY, D]

        shipyard_emb = self.cls_launch_point(shipyard_emb)
        map_emb_decode = self.cls_launch_point2(map_emb_decode)
        scores = torch.matmul(shipyard_emb, map_emb_decode.transpose(1, 2))
        # scores = self.cls_launch_point(map_emb_decode)

        # scores: [B, SY, N]

        max_shipyard = shipyard_index.shape[1]
        batch_size = shipyard_index.shape[0]

        shipyard_range = torch.tile(
            torch.arange(max_shipyard, device=scores.device), [batch_size, 1])
        scores[embs['batch_index'], shipyard_range, shipyard_index] = -1e10

        return scores
        pass

    def forward_launch_number(self, embs, shipyard_index):
        shipyard_emb = embs['map_emb'][embs['batch_index'], shipyard_index]

        scores = self.cls_launch_number(shipyard_emb)
        return scores

    def forward_convert_point(self, embs, shipyard_index):
        map_structure = embs['map_structure']
        _, map_emb_decode = self.decoder_convert(embs['map_emb'], structure_matrix=map_structure)
        # map_emb: [B, N, D]

        shipyard_emb = map_emb_decode[embs['batch_index'], shipyard_index]
        # shipyard_emb: [B, SY, D]
        shipyard_emb = self.cls_convert_point(shipyard_emb)
        map_emb_decode = self.cls_convert_point2(map_emb_decode)
        scores = torch.matmul(shipyard_emb, map_emb_decode.transpose(1, 2))
        # scores: [B, SY, N]
        max_shipyard = shipyard_index.shape[1]
        batch_size = shipyard_index.shape[0]

        shipyard_range = torch.tile(
            torch.arange(max_shipyard, device=scores.device), [batch_size, 1])
        scores[embs['batch_index'], shipyard_range, shipyard_index] = -1e10

        return scores

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
            idx = pend

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
            idx = pend

            if start > pend:
                continue
            elif end < pstart:
                break
            else:
                copy_start = max(start - pstart, 0)
                copy_end = min(psize, end - pstart)
                v_start = max(pstart - start, 0)
                v_end = min(vsize, pend - start)
                values[v_start: v_end] = param.data.flatten()[copy_start:copy_end].cpu().numpy()
                pass
            pass

        return values
        pass
    pass

