import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from .ae import PatchAutoEncoder


def load() -> torch.nn.Module:
    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, map_location="cpu", weights_only=False)


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    sign = (x >= 0).float() * 2 - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor: ...
    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor: ...


class BSQ(nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self.codebook_bits = codebook_bits
        self.embedding_dim = embedding_dim
        self.down = nn.Linear(embedding_dim, codebook_bits, bias=False)
        self.up = nn.Linear(codebook_bits, embedding_dim, bias=False)

    #@torch.amp.autocast("cuda")
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        
        orig_shape = x.shape
        if x.ndim == 4:
            x = x.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        elif x.ndim == 3:
            x = x.reshape(-1, self.embedding_dim)
        x = self.down(x)
        x = F.normalize(x, dim=-1)
        x = diff_sign(x)
        
        if len(orig_shape) == 4:
            B, C, H, W = orig_shape
            x = x.view(B, H, W, self.codebook_bits).permute(0, 3, 1, 2)
        return x

    #@torch.amp.autocast("cuda")
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.ndim == 4:
            x = x.permute(0, 2, 3, 1).reshape(-1, self.codebook_bits)
        elif x.ndim == 3:
            x = x.reshape(-1, self.codebook_bits)
        x = self.up(x)
        if len(orig_shape) == 4:
            B, _, H, W = orig_shape
            x = x.view(B, H, W, self.embedding_dim).permute(0, 3, 1, 2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        bits = (x >= 0).int()

        if bits.ndim == 4:
            bits = bits.permute(0, 2, 3, 1)  
        weights = (2 ** torch.arange(self.codebook_bits, device=x.device)).view(1, 1, 1, -1)
        return (bits * weights).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        bits = (x[..., None] & (2 ** torch.arange(self.codebook_bits, device=x.device))) > 0
        return bits.float() * 2 - 1

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self._index_to_code(x))


class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim)
        self.codebook_bits = codebook_bits
        self.bsq = BSQ(codebook_bits=codebook_bits, embedding_dim=latent_dim)

    #@torch.amp.autocast("cuda")
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).contiguous()
        latent = self.encoder(x)
        return self.bsq.encode(latent)

    #@torch.amp.autocast("cuda")
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        decoded = self.bsq.decode(x)
        x_hat = self.decoder(decoded)
        return x_hat.permute(0, 2, 3, 1).contiguous()

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).contiguous()
        latent = self.encoder(x)
        return self.bsq.encode_index(latent)

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        code = self.bsq.decode_index(x)
        x_hat = self.decoder(code)
        return x_hat.permute(0, 2, 3, 1).contiguous()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = x.permute(0, 3, 1, 2).contiguous()
        z = self.encoder(x)
        zq = self.bsq.encode(z)
        x_hat = self.decoder(self.bsq.decode(zq))
        x_hat = x_hat.permute(0, 2, 3, 1).contiguous()

        x_hat = x_hat[:, :x.shape[2], :x.shape[3], :]

        with torch.no_grad():
            indices = self.bsq._code_to_index(zq)
            cnt = torch.bincount(indices.flatten(), minlength=2 ** self.codebook_bits)
            cb0 = (cnt == 0).float().mean()
            cb2 = (cnt <= 2).float().mean()

        losses = {"cb0": cb0.detach(), "cb2": cb2.detach()}
        return x_hat, losses
