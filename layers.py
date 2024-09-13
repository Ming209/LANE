import torch
import torch.nn as nn
import torch.nn.functional as fn
import math
import copy

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(
        self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps
    ):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": fn.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
         
class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output

class TransformerEncoder(nn.Module):
    r"""One TransformerEncoder consists of several TransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class HybridAttention(nn.Module):
    """
    Hybrid Attention layer: combine time domain self-attention layer and frequency domain attention layer.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head Hybrid Attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head Hybrid Attention layer

    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps, i,  config):
        super(HybridAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.factor = config['topk_factor']
        self.scale = None
        self.mask_flag = True
        self.output_attention = False
        self.dropout = nn.Dropout(0.1)
        self.config = config
        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query_layer = nn.Linear(hidden_size, self.all_head_size)
        self.key_layer = nn.Linear(hidden_size, self.all_head_size)
        self.value_layer = nn.Linear(hidden_size, self.all_head_size)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.filter_mixer = None
        self.global_ratio = config['global_ratio']
        self.n_layers = config['n_layers']
        if self.global_ratio > (1 / self.n_layers):
            print("{}>{}:{}".format(self.global_ratio, 1 / self.n_layers, self.global_ratio > (1 / self.n_layers)))
            self.filter_mixer = 'G'
        else:
            print("{}>{}:{}".format(self.global_ratio, 1 / self.n_layers, self.global_ratio > (1 / self.n_layers)))
            self.filter_mixer = 'L'
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']
        self.dual_domain = config['dual_domain']
        self.slide_step = ((self.max_item_list_length // 2 + 1) * (1 - self.global_ratio)) // (self.n_layers - 1)
        self.local_ratio = 1 / self.n_layers
        self.filter_size = self.local_ratio * (self.max_item_list_length // 2 + 1)

        if self.filter_mixer == 'G':
            self.w = self.global_ratio
            self.s = self.slide_step

        if self.filter_mixer == 'L':
            self.w = self.local_ratio
            self.s = self.filter_size

        self.left = int(((self.max_item_list_length // 2 + 1) * (1 - self.w)) - (i * self.s))
        self.right = int((self.max_item_list_length // 2 + 1) - i * self.s)

        # random:
        # self.q_index = get_frequency_modes(50, 2, 'random')
        # self.k_index = get_frequency_modes(50, 2, 'random')
        # self.v_index = get_frequency_modes(50, 2, 'random')

        self.q_index = list(range(self.left, self.right))
        self.k_index = list(range(self.left, self.right))
        self.v_index = list(range(self.left, self.right))
        # if sample in time domain
        self.std = config['std']
        if self.std:
            self.time_q_index = self.q_index
            self.time_k_index = self.k_index
            self.time_v_index = self.v_index
        else:
            self.time_q_index = list(range(self.max_item_list_length // 2 + 1))
            self.time_k_index = list(range(self.max_item_list_length // 2 + 1))
            self.time_v_index = list(range(self.max_item_list_length // 2 + 1))

        print('modes_q={}, index_q={}'.format(len(self.q_index), self.q_index))
        print('modes_k={}, index_k={}'.format(len(self.k_index), self.k_index))
        print('modes_v={}, index_v={}'.format(len(self.v_index), self.v_index))

        if self.config['dual_domain']:
            self.spatial_ratio = self.config['spatial_ratio']

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # [256, 50, 2, 32]
        # return x.permute(0, 2, 1, 3)  # [256, 2, 50, 32]
        return x

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0) \
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query_layer(input_tensor)
        mixed_key_layer = self.key_layer(input_tensor)
        mixed_value_layer = self.value_layer(input_tensor)

        queries = self.transpose_for_scores(mixed_query_layer)
        keys = self.transpose_for_scores(mixed_key_layer)
        values = self.transpose_for_scores(mixed_value_layer)

        # B, H, L, E = query_layer.shape
        # AutoFormer
        B, L, H, E = queries.shape
        # print("qqqq", queries.shape)
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

            # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)

        # 装到采样的空盒子里
        q_fft_box = torch.zeros(B, H, E, len(self.q_index), device=q_fft.device, dtype=torch.cfloat)
        # print(box_xq_ft.shape, xq_ft.shape)
        for i, j in enumerate(self.q_index):
            q_fft_box[:, :, :, i] = q_fft[:, :, :, j]

        k_fft_box = torch.zeros(B, H, E, len(self.k_index), device=q_fft.device, dtype=torch.cfloat)
        for i, j in enumerate(self.q_index):
            k_fft_box[:, :, :, i] = k_fft[:, :, :, j]

        res = q_fft_box * torch.conj(k_fft_box)
        if self.config['use_filter']:
            # filter
            weight = torch.view_as_complex(self.complex_weight)
            res = res * weight
        box_res = torch.zeros(B, H, E, L // 2 + 1,  device=q_fft.device, dtype=torch.cfloat)
        for i, j in enumerate(self.q_index):
            box_res[:, :, :, j] = res[:, :, :, i]

        corr = torch.fft.irfft(box_res, dim=-1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        # print(V.shape)
        new_context_layer_shape = V.size()[:-2] + (self.all_head_size,)
        context_layer = V.view(*new_context_layer_shape)

        if self.dual_domain:
            # 装到采样的空盒子里
            # q
            q_fft_box = torch.zeros(B, H, E, len(self.time_q_index), device=q_fft.device, dtype=torch.cfloat)
            # print(box_xq_ft.shape, xq_ft.shape)
            for i, j in enumerate(self.time_q_index):
                q_fft_box[:, :, :, i] = q_fft[:, :, :, j]
            spatial_q = torch.zeros(B, H, E, L // 2 + 1, device=q_fft.device, dtype=torch.cfloat)
            for i, j in enumerate(self.time_q_index):
                spatial_q[:, :, :, j] = q_fft_box[:, :, :, i]

            # k
            k_fft_box = torch.zeros(B, H, E, len(self.time_k_index), device=q_fft.device, dtype=torch.cfloat)
            for i, j in enumerate(self.time_k_index):
                k_fft_box[:, :, :, i] = k_fft[:, :, :, j]
            spatial_k = torch.zeros(B, H, E, L // 2 + 1, device=k_fft.device, dtype=torch.cfloat)
            for i, j in enumerate(self.time_k_index):
                spatial_k[:, :, :, j] = k_fft_box[:, :, :, i]

            # v
            v_fft = torch.fft.rfft(values.permute(0, 2, 3, 1).contiguous(), dim=-1)
            # 装到采样的空盒子里
            v_fft_box = torch.zeros(B, H, E, len(self.time_v_index), device=v_fft.device, dtype=torch.cfloat)
            # print(box_xq_ft.shape, xq_ft.shape)
            for i, j in enumerate(self.time_v_index):
                v_fft_box[:, :, :, i] = v_fft[:, :, :, j]
            spatial_v = torch.zeros(B, H, E, L // 2 + 1, device=v_fft.device, dtype=torch.cfloat)
            for i, j in enumerate(self.time_v_index):
                spatial_v[:, :, :, j] = v_fft_box[:, :, :, i]

            queries = torch.fft.irfft(spatial_q, dim=-1)
            keys = torch.fft.irfft(spatial_k, dim=-1)
            values = torch.fft.irfft(spatial_v, dim=-1)

            queries = queries.permute(0, 1, 3, 2)
            keys = keys.permute(0, 1, 3, 2)
            values = values.permute(0, 1, 3, 2)

            attention_scores = torch.matmul(queries, keys.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            attention_scores = attention_scores + attention_mask
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            attention_probs = self.attn_dropout(attention_probs)
            qkv = torch.matmul(attention_probs, values)  # [256, 2, index, 32]
            context_layer_spatial = qkv.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer_spatial.size()[:-2] + (self.all_head_size,)
            context_layer_spatial = context_layer_spatial.view(*new_context_layer_shape)
            context_layer = (1 - self.spatial_ratio) * context_layer + self.spatial_ratio * context_layer_spatial

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class FEABlock(nn.Module):
    def __init__(
        self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps, n, config
    ):
        super(FEABlock, self).__init__()
        self.hybrid_attention = HybridAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps, n, config
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.hybrid_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output
    
class FEAEncoder(nn.Module):
    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12,
        config=None,
    ):

        super(FEAEncoder, self).__init__()
        self.n_layers = n_layers
        self.layer = nn.ModuleList()
        for n in range(self.n_layers):
            self.layer_ramp = FEABlock(n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps, n, config)
            self.layer.append(self.layer_ramp)

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
