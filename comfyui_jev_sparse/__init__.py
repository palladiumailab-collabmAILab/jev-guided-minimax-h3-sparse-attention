from .node import JevMiniMaxH3SparseAttention

NODE_CLASS_MAPPINGS = {
    "JevMiniMaxH3SparseAttention": JevMiniMaxH3SparseAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JevMiniMaxH3SparseAttention": "Jev MiniMax H3 Sparse Attention",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
