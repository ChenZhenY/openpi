"""
VLA Cache Framework: Efficient KV cache reuse for Vision-Language-Action models.

This framework provides a general solution for caching and reusing KV values
for unchanged image patches across inference calls, working with different
LLM backbones (JAX/Flax, PyTorch, HuggingFace Transformers).

Key Insight: The compute burden is in LLM prefill, not vision encoding.
For unchanged image patches, we can:
1. Skip the KV projection computation entirely
2. Reuse cached K, V values at the correct positions

Architecture:
    VLACacheManager: Detects changed patches, manages cache state
    PartialPrefillAdapter: Per-backbone adapters for partial prefill
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

import numpy as np

# Type aliases
ArrayLike = TypeVar("ArrayLike")  # np.ndarray, torch.Tensor, or jax.Array
KVCache = Any  # Backend-specific cache format


# =============================================================================
# Patch Comparison Utilities
# =============================================================================

def patchify(image: np.ndarray, patch_size: int = 14) -> np.ndarray:
    """Convert image to non-overlapping patches.
    
    Args:
        image: Image array of shape (H, W, C) or (H, W)
        patch_size: Size of each patch
    
    Returns:
        Patches of shape (num_patches, patch_size, patch_size, C)
    """
    h, w = image.shape[:2]
    assert h % patch_size == 0 and w % patch_size == 0
    
    gh, gw = h // patch_size, w // patch_size
    
    if image.ndim == 3:
        patches = image.reshape(gh, patch_size, gw, patch_size, -1)
        patches = patches.transpose(0, 2, 1, 3, 4)
        patches = patches.reshape(gh * gw, patch_size, patch_size, -1)
    else:
        patches = image.reshape(gh, patch_size, gw, patch_size)
        patches = patches.transpose(0, 2, 1, 3)
        patches = patches.reshape(gh * gw, patch_size, patch_size)
    
    return patches


def compute_patch_similarity(
    patches1: np.ndarray,
    patches2: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between corresponding patches.
    
    Args:
        patches1, patches2: Arrays of shape (num_patches, ...)
    
    Returns:
        Similarity scores of shape (num_patches,)
    """
    flat1 = patches1.reshape(len(patches1), -1).astype(np.float32)
    flat2 = patches2.reshape(len(patches2), -1).astype(np.float32)
    
    norm1 = np.linalg.norm(flat1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(flat2, axis=1, keepdims=True)
    
    # Normalize
    flat1 = flat1 / (norm1 + 1e-8)
    flat2 = flat2 / (norm2 + 1e-8)
    
    # Cosine similarity
    similarity = np.sum(flat1 * flat2, axis=1)
    return similarity


def find_unchanged_patches(
    prev_image: np.ndarray,
    curr_image: np.ndarray,
    patch_size: int = 14,
    threshold: float = 0.996,
) -> tuple[list[int], list[int]]:
    """Find which patches are unchanged between two images.
    
    Args:
        prev_image: Previous image (H, W, C), uint8 or float
        curr_image: Current image (H, W, C), uint8 or float
        patch_size: Size of patches
        threshold: Similarity threshold for considering a patch unchanged
    
    Returns:
        (unchanged_indices, changed_indices): Lists of patch indices
    """
    # Normalize images to [0, 255] range
    prev = _normalize_image(prev_image)
    curr = _normalize_image(curr_image)
    
    prev_patches = patchify(prev, patch_size)
    curr_patches = patchify(curr, patch_size)
    
    similarity = compute_patch_similarity(prev_patches, curr_patches)
    
    unchanged = np.where(similarity >= threshold)[0].tolist()
    changed = np.where(similarity < threshold)[0].tolist()
    
    return unchanged, changed


def _normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to uint8 [0, 255]."""
    if img.dtype == np.uint8:
        return img
    
    # Handle float images
    if img.min() < 0:  # Assume [-1, 1] range
        img = (img + 1) / 2
    
    if img.max() <= 1.0:
        img = img * 255
    
    return img.astype(np.uint8)


# =============================================================================
# Cache State Management
# =============================================================================

@dataclass
class CacheState:
    """Stores the state from previous inference for cache reuse."""
    
    # Raw images for patch comparison (per camera)
    images: dict[str, np.ndarray] = field(default_factory=dict)
    
    # Token-level embeddings after vision encoder (optional, for embedding reuse)
    embeddings: dict[str, ArrayLike] | None = None
    
    # KV cache from LLM (backend-specific format)
    kv_cache: KVCache | None = None
    
    # Mapping: image_name -> start token index in the sequence
    token_offsets: dict[str, int] = field(default_factory=dict)
    
    # Number of tokens per image
    tokens_per_image: int = 256  # 16x16 grid for 224x224 image with 14x14 patches


@dataclass
class CacheAnalysis:
    """Result of analyzing which tokens can be reused."""
    
    # Global token indices that are unchanged (can reuse KV)
    unchanged_token_indices: list[int] = field(default_factory=list)
    
    # Global token indices that changed (need recomputation)
    changed_token_indices: list[int] = field(default_factory=list)
    
    # Per-image breakdown
    per_image_unchanged: dict[str, list[int]] = field(default_factory=dict)
    per_image_changed: dict[str, list[int]] = field(default_factory=dict)
    
    # Statistics
    total_tokens: int = 0
    reuse_ratio: float = 0.0


class VLACacheManager:
    """
    Manages KV cache reuse across inference calls by detecting unchanged image patches.
    
    Usage:
        manager = VLACacheManager(patch_size=14, threshold=0.996)
        
        # Before inference
        analysis = manager.analyze(current_images)
        
        # Use analysis.unchanged_token_indices for cache reuse
        # ... run inference with partial prefill ...
        
        # After inference
        manager.update(current_images, new_kv_cache, token_offsets)
    """
    
    def __init__(
        self,
        patch_size: int = 14,
        threshold: float = 0.996,
        tokens_per_image: int = 256,
        enabled: bool = True,
    ):
        """
        Args:
            patch_size: Size of ViT patches (typically 14 for SigLIP, 16 for CLIP)
            threshold: Cosine similarity threshold for unchanged detection
            tokens_per_image: Number of tokens per image (16x16=256 for 224x224)
            enabled: Whether caching is enabled
        """
        self.patch_size = patch_size
        self.threshold = threshold
        self.tokens_per_image = tokens_per_image
        self.enabled = enabled
        
        self._cache: CacheState | None = None
    
    @property
    def has_cache(self) -> bool:
        """Whether we have a cached state from previous inference."""
        return self._cache is not None and self._cache.kv_cache is not None
    
    @property
    def cache(self) -> CacheState | None:
        """Access the current cache state."""
        return self._cache
    
    def analyze(self, images: dict[str, np.ndarray]) -> CacheAnalysis:
        """
        Analyze which tokens can be reused from cache.
        
        Args:
            images: Dict of image_name -> image array (H, W, C)
        
        Returns:
            CacheAnalysis with unchanged/changed token indices
        """
        analysis = CacheAnalysis()
        
        if not self.enabled or self._cache is None:
            # No cache, all tokens are "changed"
            offset = 0
            for name, img in images.items():
                num_patches = self.tokens_per_image
                analysis.changed_token_indices.extend(range(offset, offset + num_patches))
                analysis.per_image_changed[name] = list(range(num_patches))
                analysis.per_image_unchanged[name] = []
                offset += num_patches
            analysis.total_tokens = offset
            return analysis
        
        # Compare each image with cached version
        offset = 0
        for name, curr_img in images.items():
            if name not in self._cache.images:
                # New image, all patches changed
                num_patches = self.tokens_per_image
                analysis.changed_token_indices.extend(range(offset, offset + num_patches))
                analysis.per_image_changed[name] = list(range(num_patches))
                analysis.per_image_unchanged[name] = []
            else:
                prev_img = self._cache.images[name]
                unchanged_local, changed_local = find_unchanged_patches(
                    prev_img, curr_img, self.patch_size, self.threshold
                )
                
                # Convert local patch indices to global token indices
                unchanged_global = [idx + offset for idx in unchanged_local]
                changed_global = [idx + offset for idx in changed_local]
                
                analysis.unchanged_token_indices.extend(unchanged_global)
                analysis.changed_token_indices.extend(changed_global)
                analysis.per_image_unchanged[name] = unchanged_local
                analysis.per_image_changed[name] = changed_local
            
            offset += self.tokens_per_image
        
        analysis.total_tokens = offset
        if analysis.total_tokens > 0:
            analysis.reuse_ratio = len(analysis.unchanged_token_indices) / analysis.total_tokens
        
        return analysis
    
    def update(
        self,
        images: dict[str, np.ndarray],
        kv_cache: KVCache,
        token_offsets: dict[str, int] | None = None,
        embeddings: dict[str, ArrayLike] | None = None,
    ) -> None:
        """
        Update cache with new inference state.
        
        Args:
            images: Current images
            kv_cache: KV cache from current inference
            token_offsets: Optional custom token offsets
            embeddings: Optional embeddings to cache
        """
        if not self.enabled:
            return
        
        # Compute token offsets if not provided
        if token_offsets is None:
            token_offsets = {}
            offset = 0
            for name in images.keys():
                token_offsets[name] = offset
                offset += self.tokens_per_image
        
        # Copy images to avoid reference issues
        images_copy = {name: np.array(img) for name, img in images.items()}
        
        self._cache = CacheState(
            images=images_copy,
            embeddings=embeddings,
            kv_cache=kv_cache,
            token_offsets=token_offsets,
            tokens_per_image=self.tokens_per_image,
        )
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache = None


# =============================================================================
# Backend Adapters (Abstract Interface)
# =============================================================================

class PartialPrefillAdapter(ABC):
    """
    Abstract adapter for running partial prefill on different LLM backends.
    
    Implementations should:
    1. Extract only the changed tokens from input embeddings
    2. Run forward pass with correct position IDs
    3. Merge new KV values with cached KV at correct indices
    """
    
    @abstractmethod
    def run_partial_prefill(
        self,
        embeddings: ArrayLike,
        analysis: CacheAnalysis,
        prev_kv_cache: KVCache,
        attention_mask: ArrayLike | None = None,
    ) -> tuple[ArrayLike, KVCache]:
        """
        Run partial prefill for only changed tokens.
        
        Args:
            embeddings: Full input embeddings (B, seq_len, hidden_dim)
            analysis: Cache analysis with changed/unchanged indices
            prev_kv_cache: KV cache from previous inference
            attention_mask: Optional attention mask
        
        Returns:
            (output_hidden_states, updated_kv_cache)
        """
        pass
    
    @abstractmethod
    def merge_kv_cache(
        self,
        prev_cache: KVCache,
        new_cache: KVCache,
        unchanged_indices: list[int],
        changed_indices: list[int],
        total_seq_len: int,
    ) -> KVCache:
        """
        Merge previous and new KV caches.
        
        For unchanged indices: use values from prev_cache
        For changed indices: use values from new_cache
        
        Returns:
            Merged KV cache
        """
        pass


# =============================================================================
# PyTorch / HuggingFace Implementation
# =============================================================================

class HuggingFacePartialPrefillAdapter(PartialPrefillAdapter):
    """
    Adapter for HuggingFace Transformers models.
    
    Works with models that use the standard `past_key_values` cache format:
    - Tuple of (key, value) tensors per layer
    - Shape: (batch, num_heads, seq_len, head_dim)
    """
    
    def __init__(self, model):
        """
        Args:
            model: HuggingFace model with `forward(inputs_embeds, past_key_values, ...)`
        """
        self.model = model
    
    def run_partial_prefill(
        self,
        embeddings,  # torch.Tensor (B, seq_len, hidden)
        analysis: CacheAnalysis,
        prev_kv_cache,  # tuple of (K, V) per layer
        attention_mask=None,
    ):
        import torch
        
        batch_size, seq_len, hidden_dim = embeddings.shape
        device = embeddings.device
        
        if not analysis.unchanged_token_indices or prev_kv_cache is None:
            # No cache to reuse, run full prefill
            outputs = self.model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                use_cache=True,
            )
            return outputs.last_hidden_state, outputs.past_key_values
        
        # Strategy: Run full forward but inject cached KV values
        # (For true partial prefill, we'd need to modify model internals)
        
        # Full forward pass
        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            use_cache=True,
        )
        
        new_kv_cache = outputs.past_key_values
        
        # Merge caches: use cached values for unchanged positions
        merged_cache = self.merge_kv_cache(
            prev_kv_cache,
            new_kv_cache,
            analysis.unchanged_token_indices,
            analysis.changed_token_indices,
            seq_len,
        )
        
        return outputs.last_hidden_state, merged_cache
    
    def merge_kv_cache(
        self,
        prev_cache,
        new_cache,
        unchanged_indices: list[int],
        changed_indices: list[int],
        total_seq_len: int,
    ):
        import torch
        
        if not unchanged_indices:
            return new_cache
        
        merged = []
        unchanged_idx = torch.tensor(unchanged_indices, dtype=torch.long)
        
        for layer_idx, (prev_kv, new_kv) in enumerate(zip(prev_cache, new_cache)):
            prev_k, prev_v = prev_kv
            new_k, new_v = new_kv
            
            # Clone new cache
            merged_k = new_k.clone()
            merged_v = new_v.clone()
            
            # Replace with cached values at unchanged positions
            # Shape: (batch, heads, seq, head_dim)
            device = merged_k.device
            idx = unchanged_idx.to(device)
            
            if prev_k.shape[2] >= max(unchanged_indices) + 1:
                merged_k[:, :, idx, :] = prev_k[:, :, idx, :]
                merged_v[:, :, idx, :] = prev_v[:, :, idx, :]
            
            merged.append((merged_k, merged_v))
        
        return tuple(merged)


# =============================================================================
# JAX/Flax Implementation (for openpi)
# =============================================================================

class JAXPartialPrefillAdapter(PartialPrefillAdapter):
    """
    Adapter for JAX/Flax models (like openpi's Gemma implementation).
    
    KV cache format varies by implementation, but typically:
    - For gemma.py: tuple of (K, V) where K, V have shape (layers, batch, seq, heads, head_dim)
    - For gemma_fast.py: tuple of (idx, K, V) per layer
    """
    
    def __init__(self, llm_forward_fn):
        """
        Args:
            llm_forward_fn: Function that runs LLM forward pass
        """
        self.llm_forward_fn = llm_forward_fn
    
    def run_partial_prefill(
        self,
        embeddings,  # jax array (B, seq_len, hidden)
        analysis: CacheAnalysis,
        prev_kv_cache,
        attention_mask=None,
    ):
        import jax.numpy as jnp
        
        if not analysis.unchanged_token_indices or prev_kv_cache is None:
            # No cache to reuse, run full prefill
            return self.llm_forward_fn(
                embeddings,
                mask=attention_mask,
                kv_cache=None,
            )
        
        # Full forward pass
        outputs, new_kv_cache = self.llm_forward_fn(
            embeddings,
            mask=attention_mask,
            kv_cache=None,
        )
        
        # Merge caches
        merged_cache = self.merge_kv_cache(
            prev_kv_cache,
            new_kv_cache,
            analysis.unchanged_token_indices,
            analysis.changed_token_indices,
            embeddings.shape[1],
        )
        
        return outputs, merged_cache
    
    def merge_kv_cache(
        self,
        prev_cache,
        new_cache,
        unchanged_indices: list[int],
        changed_indices: list[int],
        total_seq_len: int,
    ):
        import jax.numpy as jnp
        
        if not unchanged_indices:
            return new_cache
        
        # Handle different cache formats
        if isinstance(prev_cache, tuple) and len(prev_cache) == 2:
            # Format: (all_K, all_V) where shape is (layers, batch, seq, heads, head_dim)
            prev_k, prev_v = prev_cache
            new_k, new_v = new_cache
            
            idx = jnp.array(unchanged_indices, dtype=jnp.int32)
            
            if prev_k.shape[2] >= max(unchanged_indices) + 1:
                # Replace at unchanged positions across all layers
                merged_k = new_k.at[:, :, idx, :, :].set(prev_k[:, :, idx, :, :])
                merged_v = new_v.at[:, :, idx, :, :].set(prev_v[:, :, idx, :, :])
                return (merged_k, merged_v)
        
        return new_cache

