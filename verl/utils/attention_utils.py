# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Callable

_index_first_axis, _pad_input, _rearrange, _unpad_input = None, None, None, None


def _get_attention_functions() -> tuple[Callable, Callable, Callable, Callable]:
    """Dynamically import attention functions based on available hardware."""

    from verl.utils.device import is_cuda_available, is_npu_available

    global _index_first_axis, _pad_input, _rearrange, _unpad_input

    if _index_first_axis is not None:
        return _index_first_axis, _pad_input, _rearrange, _unpad_input

    def _torch_index_first_axis(*args, **kwargs):
        # Signature compatible with flash_attn.bert_padding.index_first_axis(hidden_states, indices)
        if args:
            x = args[0]
            indices = args[1]
        else:
            x = kwargs.get("hidden_states", kwargs.get("x"))
            indices = kwargs["indices"]
        return x.index_select(0, indices)

    def _torch_unpad_input(*args, **kwargs):
        # x: (batch, seqlen, ...)
        # attention_mask: (batch, seqlen) with 1/True for tokens to keep
        import torch

        if args:
            x = args[0]
            attention_mask = args[1]
        else:
            x = kwargs.get("hidden_states", kwargs.get("x"))
            attention_mask = kwargs.get("attention_mask", kwargs.get("mask"))

        if attention_mask.dtype != torch.bool:
            mask = attention_mask != 0
        else:
            mask = attention_mask

        bsz, seqlen = mask.shape
        flat_mask = mask.reshape(-1)
        indices = flat_mask.nonzero(as_tuple=False).squeeze(-1)

        x_flat = x.reshape(bsz * seqlen, *x.shape[2:])
        x_unpad = x_flat.index_select(0, indices)

        lengths = mask.sum(dim=1, dtype=torch.int32)
        cu_seqlens = torch.zeros(bsz + 1, device=x.device, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
        max_seqlen = int(lengths.max().item()) if bsz > 0 else 0

        return x_unpad, indices, cu_seqlens, max_seqlen

    def _torch_pad_input(*args, **kwargs):
        # Signature compatible with flash_attn.bert_padding.pad_input(hidden_states, indices, batch, seqlen)
        import torch

        if args:
            x_unpad = args[0]
            indices = args[1]
            bsz = args[2]
            seqlen = args[3]
        else:
            x_unpad = kwargs.get("hidden_states", kwargs.get("x_unpad"))
            indices = kwargs["indices"]
            bsz = kwargs.get("batch", kwargs.get("bsz"))
            seqlen = kwargs.get("seqlen")

        out = torch.zeros((bsz * seqlen, *x_unpad.shape[1:]), device=x_unpad.device, dtype=x_unpad.dtype)
        out.index_copy_(0, indices, x_unpad)
        return out.reshape(bsz, seqlen, *x_unpad.shape[1:])

    def _torch_rearrange(x, pattern: str, **_kwargs):
        """
        Minimal `einops.rearrange` replacement for the patterns used in VERL:
        - "b s ... -> (b s) ..."
        - "c b s ... -> (b s) c ..."
        """
        p = pattern.replace(" ", "")
        if p == "bs...->(bs)...":
            b, s = x.shape[0], x.shape[1]
            return x.reshape(b * s, *x.shape[2:])
        if p == "cbs...->(bs)c...":
            c, b, s = x.shape[0], x.shape[1], x.shape[2]
            x = x.permute(1, 2, 0, *range(3, x.dim()))  # (b,s,c,...)
            return x.reshape(b * s, c, *x.shape[3:])
        raise RuntimeError(f"Unsupported rearrange pattern: {pattern!r}")

    def _as_bool(x) -> bool:
        return bool(x() if callable(x) else x)

    if _as_bool(is_cuda_available):
        # flash_attn may be installed but ABI-broken in some containers. Fall back if import fails.
        try:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input  # type: ignore

        except Exception:
            # Pure PyTorch fallback (doesn't touch transformers/flash_attn).
            index_first_axis, pad_input, unpad_input, rearrange = (
                _torch_index_first_axis,
                _torch_pad_input,
                _torch_unpad_input,
                _torch_rearrange,
            )
    elif _as_bool(is_npu_available):
        from verl.utils.npu_utils import index_first_axis, pad_input, rearrange, unpad_input
    else:
        index_first_axis, pad_input, unpad_input, rearrange = (
            _torch_index_first_axis,
            _torch_pad_input,
            _torch_unpad_input,
            _torch_rearrange,
        )

    _index_first_axis, _pad_input, _rearrange, _unpad_input = index_first_axis, pad_input, rearrange, unpad_input

    return _index_first_axis, _pad_input, _rearrange, _unpad_input


def index_first_axis(*args, **kwargs):
    """
    Unified entry point for `index_first_axis` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.index_first_axis`
      - On NPU: `transformers.integrations.npu_flash_attention.index_first_axis`
        (falls back to `transformers.modeling_flash_attention_utils._index_first_axis`
        in newer versions of transformers).

    Users can call this function directly without worrying about the underlying device.
    """
    func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def pad_input(*args, **kwargs):
    """
    Unified entry point for `pad_input` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.pad_input`
      - On NPU: `transformers.integrations.npu_flash_attention.pad_input`
        (falls back to `transformers.modeling_flash_attention_utils._pad_input`
        in newer versions of transformers).

    Users can call this function directly without worrying about the underlying device.
    """
    _, func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def rearrange(*args, **kwargs):
    """
    Unified entry point for `rearrange` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.rearrange`
      - On NPU: `transformers.integrations.npu_flash_attention.rearrange`
        (falls back to `einops.rearrange` if no dedicated NPU implementation exists).

    Users can call this function directly without worrying about the underlying device.
    """
    *_, func, _ = _get_attention_functions()
    return func(*args, **kwargs)


def unpad_input(*args, **kwargs):
    """
    Unified entry point for `unpad_input` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.unpad_input`
      - On NPU: `transformers.integrations.npu_flash_attention.unpad_input`
        (falls back to `transformers.modeling_flash_attention_utils._unpad_input`
        in newer versions of transformers).

    Users can call this function directly without worrying about the underlying device.
    """
    *_, func = _get_attention_functions()
    return func(*args, **kwargs)


__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input"]
