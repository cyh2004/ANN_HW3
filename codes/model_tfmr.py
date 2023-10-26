from turtle import position
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

ACT2FN = {
    "relu": F.relu,
    "tanh": torch.tanh,
    "linear": lambda x: x,
    "sigmoid": torch.sigmoid,
    "gelu": F.gelu,
}

class TransposeLinear(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        # 元组相加，将size_out改成(batch_size, num_steps, self.nf)
        size_out = x.size()[:-1] + (self.nf,)
        # addmm中 x变成二维的，(batch_size * num_steps, nx)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        # 输出的x最低维度是self.nf
        return x

class TfmrAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            # TODO START
            # define the bias term for constructing the causal mask (i.e., seeing only prefix tokens).
            torch.arange(1, max_positions + 1).reshape((1, -1))
            # TODO END
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.c_attn = TransposeLinear(3 * self.embed_dim, self.embed_dim)
        self.c_proj = TransposeLinear(self.embed_dim, self.embed_dim)

        self.wq = TransposeLinear(self.embed_dim, self.embed_dim)
        self.wk = TransposeLinear(self.embed_dim, self.embed_dim)
        self.wv = TransposeLinear(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    def _attn(self, query, key, value, past_len):
        # TODO START
        # Input Size: (batch_size, num_heads, seq_len, attn_head_size)
        # implement the multi-head mask self-attnetion mechanism  
        
        # 注意这里需要交换key的倒数第一和第二个维度
        shape = query.shape[:2]
        # Size: (batch_size*num_heads, query_len, key_len)
        attn_weights = torch.bmm(query.reshape(-1, query.shape[-2], query.shape[-1]), key.reshape(-1, key.shape[-2], key.shape[-1]).transpose(-2,-1))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        # Size: (batch_size*num_heads, num_steps)
        # valid_len = torch.arange(1, num_steps + 1).repeat((batch_size, num_heads, 1))
        valid_len = self.bias[:, :attn_weights.shape[-2]]
        causal_mask = valid_len.unsqueeze(0) < valid_len.unsqueeze(-1)
        attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))
        attn_weights = torch.nn.functional.softmax(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)
        # Size: (batch_size*num_heads, query_len, key_len)
        attn_output = torch.bmm(attn_weights, value.reshape(-1, value.shape[-2], value.shape[-1]))
        attn_output = attn_output.reshape(*shape, attn_output.shape[-2], attn_output.shape[-1])
        attn_weights = attn_weights.reshape(*shape, attn_weights.shape[-2], attn_weights.shape[-1])
        # Size: (batch_size, num_heads, query_len, value_len)
        return attn_output, attn_weights
        # TODO END

    def _split_heads(self, tensor, num_heads, attn_head_size):
        # TODO START
        # Splits hidden_size dim into attn_head_size and num_heads
        # Input Size: (batch_size, sequence_length, hidden_size)
        # Output Size: (batch_size, num_attn_heads, sequence_length, attn_head_size)
        
        # Size: (batch_size, sequence_length, num_heads, hidden_size//num_heads)
        tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], num_heads, -1)
        return tensor.permute(0, 2, 1, 3)
        # TODO END

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        # TODO START
        # Merges attn_head_size dim and num_attn_heads dim into hidden_size
        # Input Size: (batch_size, num_attn_heads, sequence_length, attn_head_size)
        # Output Size: (batch_size, sequence_length, hidden_size)
        
        tensor = tensor.permute(0, 2, 1, 3)
        
        return tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
        # TODO END

    def forward(
        self,
        hidden_states,
        layer_past=None,
        use_cache=False,
    ):
        # hidden_states 从 (batch_size, sequence_length, hidden_size) 变成 (batch_size, sequence_length, 3 * hidden_size)
        # 然后被split成 (batch_size, sequence_length, hidden_size)
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self._split_heads(self.wq(query), self.num_heads, self.head_dim)
        key = self._split_heads(self.wk(key), self.num_heads, self.head_dim)
        value = self._split_heads(self.wv(value), self.num_heads, self.head_dim)
        if layer_past is not None:
            past_key, past_value = layer_past
            past_len = past_key.size(-2)
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        else:
            past_len = 0

        if use_cache is True:
            present = (key, value)
        else:
            present = None
        
        # Size: (batch_size, num_heads, query_len, value_len)
        attn_output, attn_weights = self._attn(query, key, value, past_len)
        
        # Size: (batch_size, query_len, value_len)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class TfmrMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = TransposeLinear(intermediate_size, embed_dim)
        self.c_proj = TransposeLinear(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TfmrBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = TfmrAttention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = TfmrMLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        use_cache=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        # Size: (batch_size, num_hiddens, embed_size)
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        # 经过一次attn就加一个present
        outputs = attn_outputs[1:] # present, attn_weights

        # TODO START
        # Implement the rest of the Tranformer block (residual connection, layer norm, feedforward)
        # NOTE: We implement the Pre-Norm version of Transformer, where the ln_1 and ln_2 are place at the residual branch
        # HINT: You can refer to Page 39 in lecture 8 for more details
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        # TODO END

        if use_cache:
            # 加上present 和 attn_weights
            outputs = (hidden_states,) + outputs
        else:
            # 只加 attn_weights
            outputs = (hidden_states,) + outputs[1:]

        return outputs


class TfmrModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([TfmrBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def get_input_embeddings(self):
        return self.wte

    def forward(
        self,
        input_ids,
        past_key_values=None,
        use_cache=None,
    ):
        # input_ids 是一句话中所有token在词表中的下标
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Size: (batch_size, num_steps, embed_size)
        inputs_embeds = self.wte(input_ids)

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        # TODO START
        # Implement the positional embeddings. Note that the length of cache hidden states used during inference
        position_embeds = torch.zeros((1, inputs_embeds.shape[1], inputs_embeds.shape[2])).to(device)
        i = torch.arange(0, inputs_embeds.shape[1], dtype=torch.float32).reshape((-1, 1)).to(device)
        j = torch.arange(0, inputs_embeds.shape[2], 2, dtype=torch.float32) / inputs_embeds.shape[1]
        j = j.to(device)
        j = j.reshape((1, -1))
        i = i / torch.pow(10000, j)
        
        position_embeds[:, :, 0::2] = torch.sin(i)
        position_embeds[:, :, 1::2] = torch.cos(i)
        # TODO END
        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = ()
        all_cross_attentions = ()
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                use_cache=use_cache,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        all_hidden_states = all_hidden_states + (hidden_states,)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": presents,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attentions,
            "cross_attentions": all_cross_attentions,
        }


class TfmrLMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = TfmrModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids,
        past_key_values=None,
        labels=None,
        use_cache=None,
        PAD_ID=None,
    ):
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = transformer_outputs["last_hidden_state"]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            ce_loss_fct = CrossEntropyLoss(reduction="none")
            # TODO START
            # def cross_entropy(y_hat, y):
            #   return - np.log(y_hat[range(len(y_hat)), y])
            # lm_logits: (batch_size, num_steps, vocab_size)
            # labels: (batch_size, num_steps)
            valid_len = (labels == PAD_ID)
            # valid_len[:, 0] = False
            # valid_len: (batch_size, num_steps)
            valid_len = ~valid_len
            # loss = torch.nn.functional.softmax(lm_logits, dim=-1)
            # loss: (batch_size, num_steps)
            loss = ce_loss_fct(lm_logits.reshape(-1, lm_logits.shape[-1]), labels.reshape(-1))
            loss = loss[valid_len.reshape(-1)].mean()
            # loss = loss.mean()
            
            # Implement the loss function. Note that you should shift logits so that tokens < n predict n
            # HINT: We set the loss to 0 where [PAD] token is the label, except for the last token, where [PAD] token worked as the "eod of sentence" token.
            # TODO END

        return {
            "loss": loss,
            "logits": lm_logits,
            "past_key_values": transformer_outputs["past_key_values"],
            "hidden_states": transformer_outputs["hidden_states"],
            "attentions": transformer_outputs["attentions"],
            "cross_attentions": transformer_outputs["cross_attentions"],
         }
        

    def inference(self, device, PAD_ID, batch_size, maxlen, decode_strategy, temperature, top_p=1.0):
        self.eval()
        allgen = []
        with torch.no_grad():
            for i in range(0, int(5000/batch_size)+1):
                input_ids = torch.tensor([[PAD_ID] for _ in range(batch_size)]).to(device)
                past_key_values = None
                output_ids = input_ids
                for _ in range(maxlen):
                    outputs = self(input_ids, past_key_values=past_key_values, use_cache=True)
                    logits = outputs["logits"]
                    # 包含了3个block
                    # 每个block有key和value
                    # 每次key的倒数第二个维度，也就是num_steps，都会加一
                    # 变化过程大概是：
                    # len(past_key_values) == 3
                    # len(past_key_values[0]) == 2
                    # past_key_values[0][0].shape:
                    # torch.Size([32, 12, 1, 64])
                    # torch.Size([32, 12, 2, 64])
                    # torch.Size([32, 12, 3, 64])
                    past_key_values = outputs["past_key_values"]
                    logits = logits[:, -1, :] / temperature

                    if decode_strategy == "top-p":
                        # TODO START
                        # implement top-p sampling
                        logits = logits.softmax(dim=-1)
                        cumsum = logits.cumsum(dim=-1)
                        logits = torch.where(cumsum < top_p, logits, torch.full_like(logits, -1e4))
                        # TODO END
                    elif decode_strategy == "top_k":
                        # TODO START
                        # implement top-k sampling
                        logits = logits.softmax(dim=-1)
                        values, indices = logits.topk(10)
                        logits = torch.where(logits > values[:, -1:], logits, torch.full_like(logits, -1e4))
                        # TODO END
                    prob = logits.softmax(dim=-1) # shape: (batch_size, num_vocabs)
                    now_token = torch.multinomial(prob, 1)[:, :1] # shape: (batch_size)

                    output_ids = torch.cat([output_ids, now_token], 1)
                    input_ids = now_token
                allgen += output_ids.cpu().numpy().tolist()
        pro_allgen = []
        for gen in allgen[:5000]:
            pro_allgen.append([])
            for idx in gen[1:]:
                if idx == PAD_ID:
                    break
                pro_allgen[-1].append(idx)
        self.train() # return to training mode
        return pro_allgen
                