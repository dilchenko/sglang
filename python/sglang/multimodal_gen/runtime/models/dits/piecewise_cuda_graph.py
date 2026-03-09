import time
from dataclasses import dataclass
from typing import Any

import torch

from sglang.multimodal_gen.runtime.utils.perf_logger import (
    record_diffusion_timing_exclusion,
)


@dataclass
class PiecewiseCudaGraphEntry:
    graph: torch.cuda.CUDAGraph
    static_inputs: tuple[Any, ...]
    output: Any


class PiecewiseCudaGraphRunner:
    def __init__(self) -> None:
        self._entries: dict[tuple[Any, ...], PiecewiseCudaGraphEntry] = {}

    def _build_key(self, phase: str, *inputs: Any) -> tuple[Any, ...]:
        signatures: list[Any] = []
        self._collect_tensor_signatures(inputs, "inputs", signatures)
        signatures.sort(key=lambda item: item[0])
        return phase, tuple(signatures)

    def _collect_tensor_signatures(
        self, obj: Any, prefix: str, signatures: list[tuple[str, tuple[int, ...], str]]
    ) -> None:
        if isinstance(obj, torch.Tensor):
            signatures.append((prefix, tuple(obj.shape), str(obj.dtype)))
            return
        if isinstance(obj, dict):
            for key in sorted(obj.keys()):
                self._collect_tensor_signatures(obj[key], f"{prefix}.{key}", signatures)
            return
        if isinstance(obj, (list, tuple)):
            for idx, item in enumerate(obj):
                self._collect_tensor_signatures(item, f"{prefix}[{idx}]", signatures)

    def _clone_structure(self, obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.clone()
        if isinstance(obj, dict):
            return {key: self._clone_structure(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self._clone_structure(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(self._clone_structure(item) for item in obj)
        return obj

    def _copy_structure(self, src: Any, dst: Any) -> None:
        if isinstance(src, torch.Tensor):
            if not isinstance(dst, torch.Tensor):
                raise TypeError(
                    f"Piecewise CUDA graph static tensor mismatch: {type(dst)}"
                )
            if src.shape != dst.shape:
                raise ValueError(
                    f"Piecewise CUDA graph tensor shape mismatch: {src.shape} vs {dst.shape}"
                )
            dst.copy_(src)
            return
        if isinstance(src, dict):
            if not isinstance(dst, dict):
                raise TypeError(
                    f"Piecewise CUDA graph static dict mismatch: {type(dst)}"
                )
            for key, value in src.items():
                self._copy_structure(value, dst[key])
            return
        if isinstance(src, list):
            if not isinstance(dst, list) or len(src) != len(dst):
                raise TypeError("Piecewise CUDA graph static list mismatch")
            for src_item, dst_item in zip(src, dst, strict=True):
                self._copy_structure(src_item, dst_item)
            return
        if isinstance(src, tuple):
            if not isinstance(dst, tuple) or len(src) != len(dst):
                raise TypeError("Piecewise CUDA graph static tuple mismatch")
            for src_item, dst_item in zip(src, dst, strict=True):
                self._copy_structure(src_item, dst_item)

    def replay_or_capture(self, phase: str, fn, *inputs: Any) -> Any:
        key = self._build_key(phase, *inputs)
        entry = self._entries.get(key)
        if entry is None:
            capture_start_time = time.perf_counter()
            static_inputs = tuple(self._clone_structure(item) for item in inputs)
            _ = fn(*static_inputs)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = fn(*static_inputs)
            entry = PiecewiseCudaGraphEntry(
                graph=graph,
                static_inputs=static_inputs,
                output=output,
            )
            self._entries[key] = entry
            record_diffusion_timing_exclusion(time.perf_counter() - capture_start_time)
            return output

        for src, dst in zip(inputs, entry.static_inputs, strict=True):
            self._copy_structure(src, dst)
        entry.graph.replay()
        return entry.output


def get_padded_length(length: int, buckets: tuple[int, ...]) -> int:
    for bucket in buckets:
        if length <= bucket:
            return bucket
    return buckets[-1] * ((length + buckets[-1] - 1) // buckets[-1])


def pad_tensor_to_length(
    tensor: torch.Tensor,
    target_length: int,
    dim: int = 1,
) -> torch.Tensor:
    current_length = tensor.shape[dim]
    if current_length == target_length:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_length - current_length
    pad_tensor = tensor.new_zeros(*pad_shape)
    return torch.cat([tensor, pad_tensor], dim=dim)
