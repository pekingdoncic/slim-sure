"""Local Sobel texture and fixed native candidate-count allocation."""
import math
import torch
from torch.nn import functional as F

TEXTURE_SCHEMA = "sobel3_scale1over8_window16_offsets_minus8_plus7_valid_v1"


def local_texture(image, valid_mask, coarse_hw):
    """Return [B,Hc*Wc] energy means/counts at native zero-offset anchors.

    Input is pre-model-normalization gray [0,1], [B,1,H,W]. Sobel extends
    image boundaries by replication. At interior padding, invalid stencil
    neighbors are replaced by the valid center pixel, avoiding false edges.
    Window [-8,+7] includes only valid image pixels, never padded pixels.
    """
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError("Texture requires grayscale [B,1,H,W]")
    if valid_mask.ndim == 2:
        valid_mask = valid_mask.unsqueeze(0)
    if valid_mask.shape != image[:, 0].shape:
        raise ValueError("Texture requires the full input-resolution validity mask")
    if valid_mask.dtype != torch.bool:
        raise TypeError("Validity mask must be boolean")
    if not torch.isfinite(image).all() or (image < 0).any() or (image > 1).any():
        raise ValueError("Texture input must be finite grayscale in [0,1]")
    b, _, h, w = image.shape
    hc, wc = map(int, coarse_hw)
    if hc <= 0 or wc <= 0 or h % hc or w % wc:
        raise ValueError("Texture anchors require verified integer per-axis strides")
    with torch.autocast(device_type=image.device.type, enabled=False):
        gray = image.float()
        valid = valid_mask.unsqueeze(1)
        patches = F.unfold(F.pad(gray, (1,1,1,1), mode="replicate"), kernel_size=3).view(b,9,h,w)
        masks = F.unfold(F.pad(valid.float(), (1,1,1,1), mode="replicate"), kernel_size=3).view(b,9,h,w).bool()
        patches = torch.where(masks, patches, gray.expand(-1,9,-1,-1))
        kx = gray.new_tensor([-1,0,1,-2,0,2,-1,0,1]).view(1,9,1,1) / 8.0
        ky = gray.new_tensor([-1,-2,-1,0,0,0,1,2,1]).view(1,9,1,1) / 8.0
        energy = ((patches*kx).sum(1).square() + (patches*ky).sum(1).square()) * valid_mask
        # Integral images avoid allocating 256 samples per full-resolution pixel.
        energy_sum = F.pad(energy.cumsum(1).cumsum(2), (1,0,1,0))
        valid_sum = F.pad(valid_mask.float().cumsum(1).cumsum(2), (1,0,1,0))
        yy, xx = torch.meshgrid(torch.arange(hc, device=image.device)*(h//hc),
                                torch.arange(wc, device=image.device)*(w//wc), indexing="ij")
        y0, y1 = (yy-8).clamp(min=0), (yy+8).clamp(max=h)
        x0, x1 = (xx-8).clamp(min=0), (xx+8).clamp(max=w)
        def window_sum(integral):
            return (integral[:,y1,x1]-integral[:,y0,x1]-integral[:,y1,x0]+integral[:,y0,x0]).flatten(1)
        count = window_sum(valid_sum)
        total = window_sum(energy_sum)
        mean = torch.where(count > 0, total/count.clamp_min(1), torch.full_like(total, float("nan")))
    return mean, count


def allocate_candidates(native_ids, norms, legal_mask, texture, threshold, fraction=0.2):
    """Reserve low-texture slots and fill by native norm ranking; keep count.

    IDs are restored to ascending native grid order. Equal norms break ties by
    ascending grid ID. A zero reserve ratio returns the exact native array.
    """
    if fraction not in (0.0, 0.2):
        raise ValueError("Only preregistered fractions 0 and 0.2 are supported")
    if native_ids.ndim != 1 or norms.ndim != 1 or legal_mask.shape != norms.shape or texture.shape != norms.shape:
        raise ValueError("Candidate inputs must be aligned vectors")
    if native_ids.dtype != torch.long or legal_mask.dtype != torch.bool:
        raise TypeError("Candidate IDs must be long and legal mask boolean")
    if native_ids.numel() != native_ids.unique().numel():
        raise ValueError("Native candidates must be unique")
    if native_ids.numel() and (not legal_mask[native_ids].all()):
        raise ValueError("Native candidate is outside the legal grid")
    if fraction == 0.0 or native_ids.numel() == 0:
        return native_ids.clone()
    if threshold is None or not math.isfinite(float(threshold)):
        raise ValueError("Freeze a finite training-only texture threshold first")
    legal_ids = torch.where(legal_mask & torch.isfinite(norms))[0]
    if legal_ids.numel() < native_ids.numel():
        raise ValueError("Insufficient legal candidates for the native count")
    order = torch.argsort(norms[legal_ids], descending=True, stable=True)
    ranked = legal_ids[order]
    low = ranked[torch.isfinite(texture[ranked]) & (texture[ranked] <= threshold)]
    reserved = low[:math.floor(fraction*native_ids.numel())]
    selected = torch.zeros_like(legal_mask)
    selected[reserved] = True
    remaining = ranked[~selected[ranked]][:native_ids.numel()-reserved.numel()]
    result = torch.cat((reserved, remaining)).sort().values
    if result.numel() != native_ids.numel():
        raise RuntimeError("Texture budget changed native candidate count")
    return result
