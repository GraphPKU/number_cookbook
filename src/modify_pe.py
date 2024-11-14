from typing import Literal, get_args
from transformers import PreTrainedModel
import transformers
from transformers.utils import logging
import torch
import functools

logger = logging.get_logger(__name__)

PEType = Literal["rope", "nope", "alibi"]
PE_CHOICES: tuple[str, ...] = get_args(PEType)


class PEModifier():
    def __init__(self, pe_type: PEType, alibi_r: float = 1.0, non_continuous_pe: bool = False):
        """
        We use a Llama model but we want to test the performance of different positional embedding. Here we modify the pe in the Llama model.
        Notice that in `transformers` library, the model has a rotary_emb Module, which forward receive (x: value_states, position_ids) as input and return (cos, sin). Then the (cos, sin) will be passed to the `apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1)` function to update q and k.
        The simplest way to modify the PE is to modify the "rotary_emb" function or modify the function `transformers.models.llama.modeling_llama.apply_rotary_pos_emb`. But for some PE (like abs), it could be impossible. If so, we need to rewrite the forward function of some modules.
        """
        if pe_type not in PE_CHOICES:
            raise ValueError(f"PE type {pe_type} is not supported.")
        self.type = pe_type
        self.alibi_r = alibi_r
        self.non_continuous_pe = non_continuous_pe

    def __call__(self, model: PreTrainedModel) -> None:
        target_func = getattr(self, f"_call_{self.type}", None)
        if target_func is not None:
            return target_func(model)
        else:
            raise ValueError(f"Method `_call_{self.type}` not found.")
    
    def _call_rope(self, model: PreTrainedModel) -> None:
        """Do nothing, just return."""
        return 
    
    def _call_nope(self, model: PreTrainedModel) -> None:
        """We modify the apply_rotary_pos_emb to return the q, k directly."""
        def new_apply(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            logger.warning_once("It is EXPECTED to read this line when you try to remove the original rope in Llama.")
            return q, k
        transformers.models.llama.modeling_llama.apply_rotary_pos_emb = new_apply
        
    def _alibi_pe(self, q_len, kv_len, num_head, device, dtype, pos_ids=None):
        if pos_ids is None:
            # unsqueeze(0) to make a batchsize dim as 1
            j = torch.arange(kv_len, device=device, dtype=dtype).unsqueeze(0)
            if q_len == kv_len:
                i = torch.arange(q_len, device=device, dtype=dtype).unsqueeze(0)
            elif q_len == 1:
                i = torch.arange(kv_len, kv_len+q_len, device=device, dtype=dtype).unsqueeze(0)
            else:
                raise ValueError(f"Unexpected length, q_len: {q_len}, kv_len: {kv_len}")
        else:
            i = pos_ids.to(device=device, dtype=dtype)
            j = pos_ids.to(device=device, dtype=dtype)
        base = torch.abs(i.unsqueeze(-1) - j.unsqueeze(-2)).unsqueeze(1) # (batchsize, 1, q_len, kv_len)
        ratio = 2**(-8*(1+torch.arange(num_head, device=device, dtype=dtype))/num_head).unsqueeze(-1).unsqueeze(-1).unsqueeze(0) # (1, num_head, 1, 1)
        return (self.alibi_r * base * ratio) # (batchsize, num_head, q_len, kv_len)
        
    def _call_alibi(self, model: PreTrainedModel) -> None:
        return self._call_additive_rpe(model=model, rpe_type="alibi")
    
    def _call_additive_rpe(self, model: PreTrainedModel, rpe_type: Literal["alibi"]) -> None:
        """
        Prepare for additive relative pe. A bias will be add to the attention mask.
        In Llama implementation, a 4d (batchsize, 1, q_len, kv_len) causal mask will be added to the original score and then softmaxed.
        So we can add the additional bias to the `causal_mask` to achieve additive relative pe.
        Modify the _prepare_4d_causal_attention_mask_with_cache_position function.
        """
        # first remove rope 
        self._call_nope(model)
        
        orig_func = transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask_with_cache_position
        model.config._attn_implementation = "eager" # just for convinience
        
        def new_func(*args, **kwargs):
            causal_mask = orig_func(*args, **kwargs)
            assert len(causal_mask.shape) == 4 and causal_mask.shape[1] == 1
            # we assume that kv_len is the total length of the input sequence
            kv_len = causal_mask.shape[-1]
            q_len = causal_mask.shape[-2]
            # we assume there are two situations: q_len == kv_len (if training or generate the first token) or q_len == 1 (if generate the next token)
            assert q_len == kv_len or q_len == 1, f"Unexpected length, q_len: {q_len}, kv_len: {kv_len}"
            if self.non_continuous_pe and q_len > 1:
                # In this situation, because we need the position ids to calculate the relative pe, we cannot calculate it here. So we just return the original causal_mask.
                # If the q_len == 1, it means we are generating the next token, and we assume that the position ids must be continuous.
                # (The random position must be False in eval_tokenizer.)
                return causal_mask
            logger.warning_once("It is EXPECTED to read this line when you try to add an additive relative pe with continuous position ids or when generating.")
            if rpe_type == "alibi":
                # we add an additional bias to the causal_mask
                alibi_pe = self._alibi_pe(q_len, kv_len, model.config.num_attention_heads, causal_mask.device, causal_mask.dtype) # (bsz=1, num_head, q_len, kv_len)
            else:
                raise ValueError(f"Invalid rep_type {rpe_type}.")
            # expand the second dim to num_head
            causal_mask = causal_mask.expand(-1, model.config.num_attention_heads, -1, -1)
            causal_mask = causal_mask + alibi_pe
            return causal_mask
        
        transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask_with_cache_position = new_func
        
        if self.non_continuous_pe:
            orig_forwards = []
            for layer_idx, layer in enumerate(model.model.layers):
                orig_forwards.append(layer.forward)
                position_ids_pos = layer.forward.__code__.co_varnames.index("position_ids")
                attention_mask_pos = layer.forward.__code__.co_varnames.index("attention_mask")
                
                def new_forward(my_layer_index, *args, **kwargs):
                    if "position_ids" not in kwargs:
                        if len(args) > position_ids_pos:
                            position_ids = args[position_ids_pos]
                        else:
                            raise ValueError("The position_ids is not found.")
                    else:
                        position_ids = kwargs["position_ids"]
                        
                    if "attention_mask" not in kwargs:
                        if len(args) > attention_mask_pos:
                            attention_mask = args[attention_mask_pos]
                            from_kwargs = False
                        else:
                            raise ValueError("The attention_mask is not found.")
                    else:
                        attention_mask = kwargs["attention_mask"]
                        from_kwargs = True
                        
                    assert position_ids is not None, "The position_ids is None."
                    assert attention_mask is not None, "The attention_mask is None."
                    if position_ids.shape[-1] == 1:
                        return orig_forwards[my_layer_index](*args, **kwargs)
                    
                    # After that, we assume it is the situation that the position ids are not continuous (so the causal mask is not modified).
                    logger.warning_once(f"It is EXPECTED to read this line when you try to add an additive relative pe with **NON**-continuous position ids (in layer {my_layer_index}).")
                    # We asseme when use cache (should be in generation), the position ids are continuous. So here we do not consider the situation that use cache.
                    assert position_ids.shape[-1] == attention_mask.shape[-1] == attention_mask.shape[-2] # position_ids: (batchsize, seq_len)
                    if rpe_type == "alibi":
                        alibi = self._alibi_pe(position_ids.shape[-1], attention_mask.shape[-1], model.config.num_attention_heads, attention_mask.device, attention_mask.dtype, pos_ids=position_ids)
                        attention_mask = attention_mask.expand(-1, model.config.num_attention_heads, -1, -1)
                        attention_mask = attention_mask + alibi
                        # replace original attention_mask with the new one
                        if from_kwargs:
                            kwargs["attention_mask"] = attention_mask
                        else:
                            args = list(args)
                            args[attention_mask_pos] = attention_mask
                            args = tuple(args)
                        return orig_forwards[my_layer_index](*args, **kwargs)
                    else:
                        raise ValueError(f"Invalid rpe_type {rpe_type}")
                    
                # 
                layer.forward = functools.partial(new_forward, layer_idx)