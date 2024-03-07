import torch
from torch.nn.functional import softmax
import time
from utils.tree_infer import get_sampling_logits
from utils.misc import print_config, setup_seed, spec_stream
from utils.sampling import sample, norm_logits, max_fn

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device
):
    """
    Make causal mask used for bi-directional self-attention.
    Copied from Huggingface
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask


class Tree:
    def __init__(self, device :str = 'cpu', max_length = 512, dtype = torch.float16) -> None:
        self.tokens = torch.zeros(max_length, device=device).long()
        self.Successors :list[list[int]] = []
        self.num_nodes = 0
        self.device = device
        self.max_length = max_length
        self.dtype = dtype

    def initialize(self, attn_mask, sequence, new_tokens_buffer, parents_buffer, position_ids):
        self.full_attn_mask = attn_mask
        self.sequence = sequence
        self.new_tokens_buffer = new_tokens_buffer
        self.parents_buffer = parents_buffer
        self.position_ids = position_ids
        self.full_attn_mask = self.full_attn_mask.repeat(2, 2)

    def set_prefix(self, prefix: torch.LongTensor):
        self.tokens[:len(prefix)] = prefix.to(self.device)
        self.position_ids[:len(prefix)] = torch.arange(len(prefix))

        self.num_nodes = len(prefix)
        self.full_attn_mask[:self.max_length, :self.max_length] = _make_causal_mask((1, self.max_length),dtype=self.dtype, device=self.device)

    def collective_expand_position(self, expand_tokens :torch.LongTensor):
        self.tokens = torch.cat([self.tokens, expand_tokens], dim=-1)

    def verbose(self):
        print(self.tokens)
        print(self.Successors)

class SpecTree(Tree):
    def __init__(self, 
                 engine,
                 temperature :float = 0.6,
                 top_p: float = 0.9,
                 max_length = 256,
                 device :str = 'cpu',
                 vocab_size = 32000,
                 grow_map = None,
                 residual_graph = None,
                 sampling_callables = None,
                 sample_gather_indices = None,
                 tokenizer=None) -> None:
        
        super().__init__(device=device, max_length=max_length)
        self.graph_engine = engine
        self.temperature = temperature
        self.top_p = top_p
        self.residual_graph = residual_graph
        self.tokenizer = tokenizer
        
        # grow map
        self.grow_map = grow_map
        self.sampling_callables = sampling_callables
        self.sample_gather_indices = sample_gather_indices
        self.draft_step = len(self.grow_map["roots"])
        self.grow_map_roots_gpu = []
        for x in self.grow_map["roots"]:
            self.grow_map_roots_gpu.append(torch.Tensor(x).to(self.device).long())
        self.Successors = self.grow_map["Successors"]
        tree_mask :torch.Tensor = self.grow_map["mask"].to(self.device)
        tree_mask = (tree_mask == 0).type(self.dtype)
        tree_mask.masked_fill_(tree_mask > 0, torch.finfo(self.dtype).min)

        # initialize
        self.tree_size = self.grow_map["size"]
        self.tree_mask = tree_mask

        self.tree_mask_step = []
        self.storage_ids_step = []
        self.storage_ids = torch.arange(self.graph_engine.engine.graph_cache.max_budget, self.graph_engine.engine.graph_cache.real_budget).to(self.device)
        self.depth = self.grow_map["depth"].to(self.device)
        start = 1
        for i in range(self.draft_step - 1):
            # print(sum(grow_map['branches'][i]))
            # print(grow_map['mask'][start:start+sum(grow_map['branches'][i])])
            # print(torch.cat([torch.ones(sum(grow_map['branches'][i]), self.graph_engine.engine.graph_cache.max_budget, device='cuda:0'), tree_mask[start:start+sum(grow_map['branches'][i])]], dim=-1).shape)
            # exit()
            self.tree_mask_step.append(torch.cat([torch.zeros(sum(grow_map['branches'][i]), self.graph_engine.engine.graph_cache.max_budget, device='cuda:0'), tree_mask[start:start+sum(grow_map['branches'][i])]], dim=-1))
            self.storage_ids_step.append(self.storage_ids[start:start+sum(grow_map['branches'][i])].clone())
            start += sum(grow_map['branches'][i])

        self.draft_logits = torch.zeros((self.tree_size, vocab_size), dtype=self.dtype).to(self.device)
        self.rand = torch.empty((self.tree_size, self.draft_logits.shape[1]), dtype=self.dtype).uniform_().to(self.device)
        self.verify_tokens = torch.zeros(self.tree_size, device=device).long()
        
        #!!! VERY CONFUSED
        # print(f"Total nodes: {total_nodes}") # prefix + 64 - 1
        # print(self.full_attn_mask, self.full_attn_mask.shape)
        # print(self.attn_mask, self.attn_mask.shape)
        
        # self.ground_truth_len = len(prefix)
        # self.r = torch.rand(len(position_ids), dtype=self.dtype).to(self.device)
        
        # # print(f"Position ids: {self.position_ids[len(prefix)-10 : len(prefix) + self.tree_size - 1]}")
        # self.position_ids[len(prefix) : len(prefix) + self.tree_size - 1] = (self.grow_map["depth"][1:].to(self.device) + len(prefix) - 1)
        # # print(f"Position ids: {self.position_ids[len(prefix)-10 : len(prefix) + self.tree_size - 1]}")

    def prefill(self, prefix :torch.LongTensor):
        ##### PREFILL #####
        self.graph_engine.clear_kv()
        self.graph_engine.prefill(input_ids=prefix.unsqueeze(0)[:,:-1])
        logits = self.graph_engine.prefill(input_ids=prefix.unsqueeze(0)[:,-1:])
        next_token = sample(norm_logits(logits[:,-1,:], temperature=self.temperature ,top_k=-1, top_p=self.top_p))
        spec_stream(next_token[0], self.tokenizer, 'cyan')
        # self.graph_engine.kv_stats()
        return next_token

    def construct_grow_map(self, next_token):
        self.verify_tokens[0] = next_token
        # first feed the next token to the draft model, and get the logits
        position_ids = torch.arange(self.graph_engine.engine.kv_cache.seq_len, self.graph_engine.engine.kv_cache.seq_len+1, device=self.graph_engine.engine.model.device).unsqueeze(0)
        storage_ids = self.storage_ids[0].unsqueeze(0)
        # print(position_ids, storage_ids)
        draft_logits = self.graph_engine.inference(
            input_ids = next_token,
            position_ids = position_ids,
            attn_mask = None,
            storage_ids=storage_ids,
        )[0]
        self.draft_logits[0] = draft_logits
        for i in range(self.draft_step - 1):
            draft_logits = self.collective_grow_static(self.grow_map_roots_gpu[i], self.grow_map_roots_gpu[i+1], self.grow_map['branches'][i], grow_step=i, draft_logits=draft_logits)
            self.draft_logits[self.grow_map_roots_gpu[i+1]] = draft_logits
        
        # print(f"Draft logits: {self.draft_logits}")
        # print(f"Verify tokens: {self.verify_tokens}")

    @torch.inference_mode()
    def collective_grow_static(self, idx_list :list[int], next_idx_list, n_branch_list :list[int], grow_step = None, draft_logits=None):
        total_branch = sum(n_branch_list)

        # print(f"Grow step: {grow_step}, idx_list: {idx_list}, n_branch_list: {n_branch_list}, total_branch: {total_branch}")

        # new_tokens_set = draft_logits.topk(total_branch, dim=-1).indices
        # assert torch.allclose(self.draft_logits[idx_list], draft_logits), "Draft logits not equal"
        new_tokens_set = self.sampling_callables[grow_step](self.draft_logits[idx_list], self.rand[idx_list])
        new_tokens_set = new_tokens_set[self.sample_gather_indices[grow_step]]
        self.verify_tokens[next_idx_list] = new_tokens_set
        new_tokens_set = new_tokens_set.view(1, total_branch)
        # print(new_tokens_set)
        # assert new_tokens_set.shape == (1, total_branch), f"New tokens set shape: {new_tokens_set.shape}"

        attn_mask = self.tree_mask_step[grow_step]
        attn_mask = attn_mask[None, None, :, :]

        position_ids = (self.depth[next_idx_list] + self.graph_engine.engine.kv_cache.seq_len).unsqueeze(0)
        storage_ids = self.storage_ids_step[grow_step]
        # print(f"Position ids: {position_ids}", "Storage ids", storage_ids)
        
        draft_logits = self.graph_engine.inference(
            input_ids = new_tokens_set,
            storage_ids=storage_ids,
            position_ids=position_ids,
            attn_mask=attn_mask
        )[0]

        return draft_logits

    @torch.inference_mode()
    def accept_step(self, logits_id :int):
        p = self.target_logits[logits_id]
        draft_logits = self.draft_logits[logits_id]
        children = self.Successors[logits_id]
        
        if len(children) == 0:
            return (-1, p)
        
        for pos in children:
            token = self.verify_tokens[pos]
            q = softmax(draft_logits / self.temperature, dim=-1)
            r = torch.rand(1, device=self.graph_engine.engine.model.device)
            
            if p[token] >= r * q[token]:
                return (pos, None)
            else:
                p = self.residual_graph(p, q)
                draft_logits[token] = torch.finfo(self.dtype).min
        return (-1, p)

    @torch.inference_mode()
    def verify(self):
        # print(self.verify_tokens.shape, self.graph_engine.engine.kv_cache.seq_len)
        position_ids = (self.depth + self.graph_engine.engine.kv_cache.seq_len).unsqueeze(0)
        # print(position_ids)
        attn_mask = torch.cat([torch.zeros(self.tree_size, self.graph_engine.engine.kv_cache.seq_len, device='cuda:0'), self.tree_mask], dim=-1)[None, None, :, :]
        # print(attn_mask.shape, attn_mask)

        offset = self.graph_engine.engine.kv_cache.seq_len
        self.target_logits = self.graph_engine.verify(input_ids = self.verify_tokens.unsqueeze(0), position_ids=position_ids, attention_mask=attn_mask)[0]
        # print(f"Target logits: {self.target_logits.shape}")

        self.target_logits = get_sampling_logits(logits=self.target_logits, top_p=self.top_p, T=self.temperature, replicate=False)
        self.target_logits = softmax(self.target_logits / self.temperature, dim=-1)
        
        # print(f"Target logits: {self.target_logits.shape}, Draft logits: {self.draft_logits.shape}")

        accept_list = []
        accept_list.append(0)
        terminal = False
        while True:
            pos, res = self.accept_step(logits_id=accept_list[-1])
            if pos != -1:
                # accept
                accept_list.append(pos)
                spec_stream(self.verify_tokens[pos], self.tokenizer, 'green')
                # eos
                if self.verify_tokens[pos] == 0 or self.verify_tokens[pos] == 2:
                    terminal = True
                    break
            else:
                # reject
                residual = res
                break
        
        if not terminal:
            if torch.isnan(residual).any():
                terminal = True
            else:
                next_token = residual.multinomial(num_samples=1, replacement=True)
                spec_stream(next_token[0], self.tokenizer, 'red')
            
        # print(f"Accept list: {accept_list}, Terminal: {terminal}, Accept length: {accept_length}")
        # print(self.verify_tokens)
        # print(self.verify_tokens[accept_list])

        accept_list = accept_list[1:]
        accept_tokens = self.verify_tokens[accept_list]
        accept_tokens = torch.cat([accept_tokens, next_token], dim=-1)

        # print(offset, self.graph_engine.engine.kv_cache.seq_len)
        self.graph_engine.engine.kv_cache.gather_kv_incremental(accept_list, offset)
        # print(offset, self.graph_engine.engine.kv_cache.seq_len, accept_list, accept_tokens)
        # print(f"Next token: {next_token}")
        # exit()
        self.graph_engine.update_graph_cache()
        self.draft_logits.zero_()
        self.verify_tokens.zero_()
        self.rand = torch.empty((self.tree_size, self.draft_logits.shape[1]), dtype=self.dtype).uniform_().to(self.device)
        return next_token
