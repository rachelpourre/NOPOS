import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ae import PatchAutoEncoder


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """Differentiable sign with straight-through estimator."""
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor: ...
    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor: ...


class BSQ(nn.Module):
    """Binary Sign Quantization layer."""
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self.codebook_bits = codebook_bits
        self.embedding_dim = embedding_dim
        self.down = nn.Linear(embedding_dim, codebook_bits, bias=False)
        self.up = nn.Linear(codebook_bits, embedding_dim, bias=False)

    @torch.cuda.amp.autocast()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = F.normalize(x, dim=-1)
        x = diff_sign(x)
        return x

    @torch.cuda.amp.autocast()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    # --- Bitwise index helpers ---
    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        # Convert -1/1 bits to integers
        bits = (x >= 0).int()
        return (bits * (2 ** torch.arange(self.codebook_bits, device=x.device))).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self.codebook_bits, device=x.device))) > 0).float() - 1

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self._index_to_code(x))

class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim)
        self.codebook_bits = codebook_bits
        self.bsq = BSQ(codebook_bits, latent_dim)

    @torch.no_grad()
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast():
            latent = self.encoder(x)
            tokens = self.bsq.encode_index(latent)
        return tokens

    @torch.no_grad()
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast():
            code = self.bsq.decode_index(x)
            recon = self.decoder(code)
        return recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.bsq.encode(latent)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        decoded = self.bsq.decode(x)
        return self.decoder(decoded)

    def forward(self, x: torch.Tensor):
        """Reconstruct + monitor codebook usage."""
        latent = self.encoder(x)
        code = self.bsq.encode(latent)
        recon = self.decoder(self.bsq.decode(code))

        # Optional diagnostics: monitor code usage
        with torch.no_grad():
            idx = self.bsq._code_to_index(code)
            cnt = torch.bincount(idx.flatten(), minlength=2 ** self.codebook_bits)
            stats = {
                "cb0": (cnt == 0).float().mean(),
                "cb2": (cnt <= 2).float().mean(),
                "entropy": (-((cnt / cnt.sum() + 1e-8) * torch.log2(cnt / cnt.sum() + 1e-8))).sum(),
            }

        return recon, stats
