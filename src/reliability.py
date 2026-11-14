"""Optional FP32 coarse-parent correctness scoring; no matching decisions."""
import torch
from torch import nn

SCHEMA = "slim_coarse_reliability_v1"


class ReliabilityHead(nn.Module):
    """Shared descriptor encoding followed by the fixed 259 -> 64 -> 1 head."""
    def __init__(self, descriptor_dim):
        super().__init__()
        self.descriptor_dim = int(descriptor_dim)
        self.projection = nn.Linear(self.descriptor_dim, 64)
        self.normalization = nn.LayerNorm(64)
        self.predictor = nn.Sequential(nn.Linear(259, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, descriptor0, descriptor1, coarse_confidence):
        if descriptor0.shape != descriptor1.shape or descriptor0.shape[-1] != self.descriptor_dim:
            raise ValueError("Expected matching raw descriptor tensors [..., C]")
        if coarse_confidence.shape != descriptor0.shape[:-1]:
            raise ValueError("Confidence must align with the coarse parent axes")
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise TypeError("Reliability parameters must remain FP32")
        with torch.autocast(device_type=descriptor0.device.type, enabled=False):
            d0, d1 = descriptor0.float(), descriptor1.float()
            z0 = self.normalization(self.projection(d0))
            z1 = self.normalization(self.projection(d1))
            features = torch.cat((z0, z1, (z0-z1).abs(), z0*z1,
                                  torch.log1p(torch.linalg.vector_norm(d0, dim=-1, keepdim=True)),
                                  torch.log1p(torch.linalg.vector_norm(d1, dim=-1, keepdim=True)),
                                  coarse_confidence.float().unsqueeze(-1)), dim=-1)
            return self.predictor(features).squeeze(-1)

    def score_data(self, data):
        """Call outside backbone no_grad for training; attach scores only at inference."""
        logits = self(data["coarse_descriptor0"], data["coarse_descriptor1"],
                      data["coarse_confidence"])
        data["reliability_schema"] = SCHEMA
        data["reliability_logit"] = logits
        data["reliability_score"] = logits.sigmoid()
        if "m_idx_it" in data:
            data["fine_parent_id"] = data["coarse_match_id"][data["m_idx_it"]]
            data["fine_parent_reliability_score"] = data["reliability_score"][data["m_idx_it"]]
        return logits


def build_reliability_head(descriptor_dim, architecture="shared64_v1"):
    if architecture != "shared64_v1":
        raise ValueError("Only the preregistered shared64_v1 architecture is supported")
    return ReliabilityHead(descriptor_dim)
