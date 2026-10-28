"""Pinned CheMeleon encoder used by the Scigantic arm-C component."""

from pathlib import Path

from .download import sha256_file

ENCODER_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
ENCODER_SHA256 = "c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651"


def encoder_identity(path):
    digest = sha256_file(Path(path))
    if digest != ENCODER_SHA256:
        raise ValueError("Unrecognized or changed CheMeleon checkpoint")
    return {"kind": "chemeleon", "sha256": digest, "source_url": ENCODER_URL}


def transfer_state(module, state):
    """Require an exact encoder tensor match; no silent partial initialization."""
    import torch

    target = module.state_dict()
    if set(state) != set(target):
        raise ValueError("Encoder tensor keys differ")
    if any(
        not isinstance(state[k], torch.Tensor)
        or state[k].shape != target[k].shape
        or not torch.isfinite(state[k]).all()
        for k in target
    ):
        raise ValueError("Encoder tensor shape or values differ")
    module.load_state_dict(state, strict=True)
    return {"tensors": len(target), "parameters": sum(x.numel() for x in state.values())}


def initialize_encoder(module, path):
    import torch

    identity = encoder_identity(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    result = transfer_state(module, checkpoint["state_dict"])
    return {
        **identity,
        **result,
        "foundation_depth": checkpoint["hyper_parameters"]["depth"],
        "adapted_depth": module.depth,
    }
