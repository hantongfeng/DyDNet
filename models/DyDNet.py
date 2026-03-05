import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List

from models.modules import TimeEncoder
from utils.utils import NeighborSampler

class DyDNet(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, num_layers: int = 2, num_heads: int = 8,
                 dropout: float = 0.1, max_input_sequence_length: int = 128, device: str = 'cpu'):
        """
        FreeDyG with multi-transform enhancement (FFT, Wavelet, DCT)
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param channel_embedding_dim: int, dimension of each channel embedding
        :param num_layers: int, number of DNN layers
        :param num_heads: int, not used in DNN version but kept for compatibility
        :param dropout: float, dropout rate
        :param max_input_sequence_length: int, maximal length of the input sequence for each node
        :param device: str, device
        """
        super(DyDNet, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.device = device

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim, parameter_requires_grad=False)
        
        self.nif_feat_dim = self.channel_embedding_dim
        self.nif_encoder = NIFEncoder(nif_feat_dim=self.nif_feat_dim, device=self.device)

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.node_feat_dim, out_features=self.edge_feat_dim, bias=True),
            'edge': nn.Linear(in_features=self.edge_feat_dim, out_features=self.edge_feat_dim, bias=True),
            'time': nn.Linear(in_features=self.time_feat_dim, out_features=self.edge_feat_dim, bias=True),
            'nif': nn.Linear(in_features=self.nif_feat_dim, out_features=self.edge_feat_dim, bias=True)
        })
        self.reduce_layer = nn.Linear(4 * self.edge_feat_dim, self.edge_feat_dim)
       
        self.multi_transform_dnn_layers = nn.ModuleList([
            MultiTransformDNNLayer(
                d_model=self.edge_feat_dim,
                max_seq_len=max_input_sequence_length,
                dropout=dropout,
                device=device
            )
            for _ in range(self.num_layers)
        ])
        

        self.final_aggregation = DNNAggregation(self.edge_feat_dim, dropout)
        self.output_norm = nn.LayerNorm(self.edge_feat_dim)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        compute source and destination node temporal embeddings with multi-transform enhancement
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :return:
        """

        src_nodes_neighbor_ids, src_nodes_edge_ids, src_nodes_neighbor_times = \
           self.neighbor_sampler.get_historical_neighbors(node_ids=src_node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=self.max_input_sequence_length)

        dst_nodes_neighbor_ids, dst_nodes_edge_ids, dst_nodes_neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=dst_node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=self.max_input_sequence_length)


        src_nodes_nif_features, dst_nodes_nif_features = \
            self.nif_encoder(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids,
                           src_nodes_neighbor_ids=src_nodes_neighbor_ids,
                           dst_nodes_neighbor_ids=dst_nodes_neighbor_ids)

   
        src_features = self._get_multimodal_features(
            node_interact_times, src_nodes_neighbor_ids, src_nodes_edge_ids, 
            src_nodes_neighbor_times, src_nodes_nif_features
        )
        
        dst_features = self._get_multimodal_features(
            node_interact_times, dst_nodes_neighbor_ids, dst_nodes_edge_ids, 
            dst_nodes_neighbor_times, dst_nodes_nif_features
        )


        for multi_transform_dnn in self.multi_transform_dnn_layers:
            src_features = multi_transform_dnn(src_features)
            dst_features = multi_transform_dnn(dst_features)


        src_embeddings = self.final_aggregation(src_features)
        dst_embeddings = self.final_aggregation(dst_features)


        src_embeddings = self.output_norm(src_embeddings)
        dst_embeddings = self.output_norm(dst_embeddings)

        return src_embeddings, dst_embeddings

    def _get_multimodal_features(self, node_interact_times, nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times, nodes_nif_features):
        nodes_neighbor_node_raw_features, nodes_edge_raw_features, nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, 
                            nodes_neighbor_ids=nodes_neighbor_ids,
                            nodes_edge_ids=nodes_edge_ids, 
                            nodes_neighbor_times=nodes_neighbor_times, 
                            time_encoder=self.time_encoder)

        node_features = self.projection_layer['node'](nodes_neighbor_node_raw_features)
        edge_features = self.projection_layer['edge'](nodes_edge_raw_features)
        time_features = self.projection_layer['time'](nodes_neighbor_time_features)
        nif_features = self.projection_layer['nif'](nodes_nif_features)

        combined_features = torch.cat([node_features, edge_features, time_features, nif_features], dim=-1)
        combined_features = self.reduce_layer(combined_features)

        return combined_features

    def get_features(self, node_interact_times: np.ndarray, nodes_neighbor_ids: np.ndarray, nodes_edge_ids: np.ndarray,
                     nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        """
        get node, edge and time features
        """
        nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)]
        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(nodes_edge_ids)]
        nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - nodes_neighbor_times).float().to(self.device))
        
        nodes_neighbor_time_features[torch.from_numpy(nodes_neighbor_ids == 0)] = 0.0
        
        return nodes_neighbor_node_raw_features, nodes_edge_raw_features, nodes_neighbor_time_features

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """set neighbor sampler"""
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class MultiTransformDNNLayer(nn.Module):
    
    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1, device: str = 'cpu'):
        super(MultiTransformDNNLayer, self).__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.device = device
        
        
        self.fft_filter = FFTFilter(d_model, max_seq_len, device)
        self.wavelet_filter = WaveletFilter(d_model, max_seq_len, device)
        self.dct_filter = DCTFilter(d_model, max_seq_len, device)
        
        
        self.fusion_network = TransformFusionNetwork(
            d_model=d_model,
            num_transforms=3,  
            fusion_method='attention',  
            device=device
        )
        
        
        self.sequence_processor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        
       
        self.temporal_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=1)
        
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        """
        :param x: Tensor, shape (batch_size, seq_len, d_model)
        :return: Tensor, shape (batch_size, seq_len, d_model)
        """
        
        fft_enhanced = self.fft_filter(x)
        wavelet_enhanced = self.wavelet_filter(x)
        dct_enhanced = self.dct_filter(x)
        
        
        transform_outputs = [fft_enhanced, wavelet_enhanced, dct_enhanced]
        fused_enhanced = self.fusion_network(transform_outputs)
        
        x = self.norm1(x + fused_enhanced)
        
        x_conv = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        temporal_features = self.temporal_conv(x_conv).transpose(1, 2)  # (batch_size, seq_len, d_model)
        
        sequence_output = self.sequence_processor(x + temporal_features)
        x = self.norm2(x + sequence_output)
        
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)
        
        return x


class FFTFilter(nn.Module):
    
    def __init__(self, d_model: int, max_seq_len: int, device: str = 'cpu'):
        super(FFTFilter, self).__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.device = device
        
        
        freq_len = max_seq_len // 2 + 1
        self.freq_weight_real = nn.Parameter(torch.randn(1, freq_len, d_model) * 0.1)
        self.freq_weight_imag = nn.Parameter(torch.randn(1, freq_len, d_model) * 0.1)
        
        self.filter_control = nn.Parameter(torch.tensor(0.5))
        
        self.feature_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.shape
        
        if seq_len < self.max_seq_len:
            x_padded = F.pad(x, (0, 0, 0, self.max_seq_len - seq_len))
        else:
            x_padded = x[:, :self.max_seq_len, :]
        
        x_freq = torch.fft.rfft(x_padded, dim=1, norm='ortho')
        freq_len = x_freq.shape[1]
        
        complex_weight = torch.complex(
            self.freq_weight_real[:, :freq_len, :], 
            self.freq_weight_imag[:, :freq_len, :]
        )
        
        freq_indices = torch.arange(freq_len, device=self.device).float() / freq_len
        filter_alpha = torch.sigmoid(self.filter_control)
        
        low_pass_weight = torch.exp(-freq_indices.pow(2) * 4)
        high_pass_weight = 1 - low_pass_weight
        adaptive_weight = filter_alpha * high_pass_weight + (1 - filter_alpha) * low_pass_weight
        adaptive_weight = adaptive_weight.view(1, -1, 1).expand(-1, -1, d_model)
        
        filtered_freq = x_freq * complex_weight * adaptive_weight
        
        filtered_x = torch.fft.irfft(filtered_freq, dim=1, norm='ortho')
        filtered_x = filtered_x[:, :seq_len, :]
        
        transformed = self.feature_transform(filtered_x)
        
        return transformed


class WaveletFilter(nn.Module):
    
    def __init__(self, d_model: int, max_seq_len: int, device: str = 'cpu'):
        super(WaveletFilter, self).__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.device = device
        
        self.multi_scale_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=k//2, groups=d_model)
            for k in [3, 5, 7, 9]  
        ])
        
        self.scale_weights = nn.Parameter(torch.ones(4, d_model) * 0.25)
        
        self.feature_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor):
        """
        :param x: Tensor, shape (batch_size, seq_len, d_model)
        :return: Tensor, shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        x_conv = x.transpose(1, 2)
        
        multi_scale_outputs = []
        for i, conv in enumerate(self.multi_scale_convs):
            scale_output = conv(x_conv)  # (batch_size, d_model, seq_len)
            multi_scale_outputs.append(scale_output)
        
        multi_scale_outputs = torch.stack(multi_scale_outputs, dim=0)  # (num_scales, batch_size, d_model, seq_len)
        
        scale_weights = torch.softmax(self.scale_weights, dim=0)  # (num_scales, d_model)
        scale_weights = scale_weights.view(4, 1, d_model, 1)  # (num_scales, 1, d_model, 1)
        
        weighted_output = (multi_scale_outputs * scale_weights).sum(dim=0)  # (batch_size, d_model, seq_len)
        
        weighted_output = weighted_output.transpose(1, 2)
        
        transformed = self.feature_transform(weighted_output)
        
        gate_weights = self.gate(torch.mean(x, dim=1, keepdim=True))
        transformed = transformed * gate_weights
        
        return transformed


class DCTFilter(nn.Module):
    
    def __init__(self, d_model: int, max_seq_len: int, device: str = 'cpu'):
        super(DCTFilter, self).__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.device = device
        
        self.register_buffer('dct_matrix', self._create_dct_matrix(max_seq_len))
        
        self.freq_weights = nn.Parameter(torch.ones(max_seq_len, d_model) * 0.1)
        
        self.freq_balance = nn.Parameter(torch.tensor(0.7))
        
        self.feature_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
    def _create_dct_matrix(self, N):
        dct_matrix = torch.zeros(N, N)
        for k in range(N):
            for n in range(N):
                if k == 0:
                    dct_matrix[k, n] = math.sqrt(1/N)
                else:
                    dct_matrix[k, n] = math.sqrt(2/N) * math.cos(math.pi * k * (2*n + 1) / (2*N))
        return dct_matrix
        
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.shape
        
        if seq_len < self.max_seq_len:
            x_padded = F.pad(x, (0, 0, 0, self.max_seq_len - seq_len))
        else:
            x_padded = x[:, :self.max_seq_len, :]
        
        x_dct = torch.matmul(self.dct_matrix, x_padded)
        
        weighted_dct = x_dct * self.freq_weights.unsqueeze(0)
        
        freq_indices = torch.arange(self.max_seq_len, device=self.device).float()
        low_freq_mask = torch.exp(-freq_indices / (self.max_seq_len * 0.3))
        high_freq_mask = 1 - low_freq_mask
        
        balance = torch.sigmoid(self.freq_balance)
        freq_mask = balance * low_freq_mask + (1 - balance) * high_freq_mask
        freq_mask = freq_mask.view(-1, 1).expand(-1, d_model)
        
        filtered_dct = weighted_dct * freq_mask.unsqueeze(0)
        
        enhanced = torch.matmul(self.dct_matrix.T, filtered_dct)
        enhanced = enhanced[:, :seq_len, :]
        
        transformed = self.feature_transform(enhanced)
        
        return transformed


class TransformFusionNetwork(nn.Module):
    
    def __init__(self, d_model: int, num_transforms: int = 3, 
                 fusion_method: str = 'attention', device: str = 'cpu'):
        """
        :param d_model: 
        :param num_transforms: （FFT, Wavelet, DCT）
        :param fusion_method:  ['add', 'weighted_sum', 'attention', 'gated']
        :param device: 
        """
        super(TransformFusionNetwork, self).__init__()
        
        self.d_model = d_model
        self.num_transforms = num_transforms
        self.fusion_method = fusion_method
        self.device = device
        
        if fusion_method == 'weighted_sum':
            self.transform_weights = nn.Parameter(torch.ones(num_transforms) / num_transforms)
            
        elif fusion_method == 'attention':
            self.attention_proj = nn.Linear(d_model, 1)
            
        elif fusion_method == 'gated':
            self.gate_networks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model // 4),
                    nn.GELU(),
                    nn.Linear(d_model // 4, d_model),
                    nn.Sigmoid()
                )
                for _ in range(num_transforms)
            ])
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, transform_outputs: List[torch.Tensor]):
        """
        :param transform_outputs: List of Tensors, each shape (batch_size, seq_len, d_model)
        :return: Tensor, shape (batch_size, seq_len, d_model)
        """
        if len(transform_outputs) == 1:
            return self.fusion_layer(transform_outputs[0])
        
        if self.fusion_method == 'add':
            fused = sum(transform_outputs) / len(transform_outputs)
            
        elif self.fusion_method == 'weighted_sum':
            weights = torch.softmax(self.transform_weights, dim=0)
            fused = sum(w * out for w, out in zip(weights, transform_outputs))
            
        elif self.fusion_method == 'attention':
            scores = []
            for output in transform_outputs:
                score = self.attention_proj(output)  # (batch_size, seq_len, 1)
                scores.append(score)
            
            scores = torch.cat(scores, dim=-1)  # (batch_size, seq_len, num_transforms)
            weights = torch.softmax(scores, dim=-1)  
            
            
            fused = sum(transform_outputs[i] * weights[:, :, i:i+1] 
                       for i in range(len(transform_outputs)))
                       
        elif self.fusion_method == 'gated':
            gated_outputs = []
            for i, (output, gate_net) in enumerate(zip(transform_outputs, self.gate_networks)):
                gate = gate_net(output)
                gated_outputs.append(output * gate)
            fused = sum(gated_outputs) / len(gated_outputs)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        fused = self.fusion_layer(fused)
        
        return fused


class DNNAggregation(nn.Module):
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super(DNNAggregation, self).__init__()
        
        self.d_model = d_model
        
        self.aggregation_net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.weight_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Softmax(dim=1)
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor):
        """
        :param x: Tensor, shape (batch_size, seq_len, d_model)
        :return: Tensor, shape (batch_size, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        processed = self.aggregation_net(x)  # (batch_size, seq_len, d_model)
        
        weights = self.weight_net(processed)  # (batch_size, seq_len, 1)
        
        weighted_sum = torch.sum(processed * weights, dim=1)  # (batch_size, d_model)
        
        output = self.output_proj(weighted_sum)  # (batch_size, d_model)
        
        return output


class NIFEncoder(nn.Module):

    def __init__(self, nif_feat_dim: int, device: str = 'cpu'):
        super(NIFEncoder, self).__init__()

        self.nif_feat_dim = nif_feat_dim
        self.device = device

        self.nif_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.nif_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.nif_feat_dim, out_features=self.nif_feat_dim))

    def count_nodes_appearances(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, src_nodes_neighbor_ids: np.ndarray, dst_nodes_neighbor_ids: np.ndarray):
        src_nodes_appearances, dst_nodes_appearances = [], []
        
        for i in range(len(src_node_ids)):
            src_node_id = src_node_ids[i]
            dst_node_id = dst_node_ids[i]
            src_node_neighbor_ids = src_nodes_neighbor_ids[i]
            dst_node_neighbor_ids = dst_nodes_neighbor_ids[i]

            src_unique_keys, src_inverse_indices, src_counts = np.unique(src_node_neighbor_ids, return_inverse=True, return_counts=True)
            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(dst_node_neighbor_ids, return_inverse=True, return_counts=True)

            src_mapping_dict = dict(zip(src_unique_keys, src_counts))
            dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))

            if src_node_id in dst_mapping_dict:
                src_count_in_dst = dst_mapping_dict[src_node_id]
                src_mapping_dict[src_node_id] = src_count_in_dst
                dst_mapping_dict[src_node_id] = src_count_in_dst
            if dst_node_id in src_mapping_dict:
                dst_count_in_src = src_mapping_dict[dst_node_id]
                src_mapping_dict[dst_node_id] = dst_count_in_src
                dst_mapping_dict[dst_node_id] = dst_count_in_src

            src_node_neighbor_counts_in_dst = torch.tensor([dst_mapping_dict.get(neighbor_id, 0) for neighbor_id in src_node_neighbor_ids]).float().to(self.device)
            dst_node_neighbor_counts_in_src = torch.tensor([src_mapping_dict.get(neighbor_id, 0) for neighbor_id in dst_node_neighbor_ids]).float().to(self.device)

            src_nodes_appearances.append(torch.stack([torch.from_numpy(src_counts[src_inverse_indices]).float().to(self.device), src_node_neighbor_counts_in_dst], dim=1))
            dst_nodes_appearances.append(torch.stack([dst_node_neighbor_counts_in_src, torch.from_numpy(dst_counts[dst_inverse_indices]).float().to(self.device)], dim=1))

        src_nodes_appearances = torch.stack(src_nodes_appearances, dim=0)
        dst_nodes_appearances = torch.stack(dst_nodes_appearances, dim=0)

        return src_nodes_appearances, dst_nodes_appearances

    def forward(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, src_nodes_neighbor_ids: np.ndarray, dst_nodes_neighbor_ids: np.ndarray):
        src_nodes_appearances, dst_nodes_appearances = self.count_nodes_appearances(src_node_ids=src_node_ids,dst_node_ids=dst_node_ids,src_nodes_neighbor_ids=src_nodes_neighbor_ids, dst_nodes_neighbor_ids=dst_nodes_neighbor_ids)

        src_nodes_nif_features = self.nif_encode_layer(src_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        dst_nodes_nif_features = self.nif_encode_layer(dst_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        
        return src_nodes_nif_features, dst_nodes_nif_features
