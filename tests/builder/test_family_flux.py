"""Unit tests for FLUX family plugin and preprocessor serialization."""

from __future__ import annotations

import json
import struct
import sys
import types

import numpy as np
import pytest

try:
    from trtf_build.config import ModelConfig
    import trtf_build.families.flux as flux_mod
except (ImportError, ModuleNotFoundError):
    pytest.skip("trtf_build requires tensorrt", allow_module_level=True)


def _cfg(**raw_overrides: object) -> ModelConfig:
    payload = {
        "model_type": "flux",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "image_height": 80,
        "image_width": 96,
    }
    payload.update(raw_overrides)
    return ModelConfig.from_json(json.dumps(payload))


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _decode_blob(blob: bytes) -> tuple[dict[str, dict], bytes]:
    idx_len = struct.unpack("<I", blob[:4])[0]
    index = json.loads(blob[4:4 + idx_len].decode("utf-8"))
    payload = blob[4 + idx_len:]
    return index, payload


def test_matches_and_build_engine_not_supported() -> None:
    """Intent: verify FLUX model aliases and explicit build_engine rejection.

    Preconditions: plugin object is imported.
    Postconditions: aliases match and build_engine raises NotImplementedError.
    """
    plugin = flux_mod.plugin
    assert plugin.matches("flux")
    assert plugin.matches("flux.1")
    assert plugin.matches("flux.2")
    assert plugin.matches("flux_t2i")
    assert not plugin.matches("wan_t2v")

    with pytest.raises(NotImplementedError, match="build_components"):
        plugin.build_engine(_cfg(), {}, 16)


def test_load_weights_reads_optional_dirs_and_transformer_config(tmp_path) -> None:
    """Intent: cover load_weights success branch with optional subdirs and config JSON.

    Preconditions: model_index.json, transformer/config.json, and selected subdirs exist.
    Postconditions: output WeightDict includes discovered directories and parsed transformer config.
    """
    model_dir = tmp_path / "flux"
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "text_encoder").mkdir(parents=True)
    (model_dir / "vae").mkdir(parents=True)
    (model_dir / "model_index.json").write_text("{}")
    (model_dir / "transformer" / "config.json").write_text(
        json.dumps({"num_attention_heads": 3, "guidance_embeds": True})
    )

    weights = flux_mod.plugin.load_weights(str(model_dir), _cfg())

    assert weights["_model_format"] == "diffusers"
    assert "_text_encoder_dir" in weights
    assert "_text_encoder_2_dir" not in weights
    assert weights["_transformer_config"]["num_attention_heads"] == 3
    assert weights["_vae_dir"].endswith("vae")


def test_load_weights_rejects_missing_model_index(tmp_path) -> None:
    """Intent: cover load_weights failure branch for non-diffusers paths.

    Preconditions: directory exists without model_index.json.
    Postconditions: load_weights raises ValueError.
    """
    bad_dir = tmp_path / "flux_bad"
    bad_dir.mkdir()
    with pytest.raises(ValueError, match="Expected diffusers format"):
        flux_mod.plugin.load_weights(str(bad_dir), _cfg())


def test_build_components_with_clip_and_second_t5(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Intent: validate CLIP+T5 orchestration and derived image-token count.

    Preconditions: builder modules are replaced by deterministic stubs and both text encoders exist.
    Postconditions: both encoder plans are returned, denoiser/vae builders are called with derived args.
    """
    calls: dict[str, object] = {}

    model_dir = tmp_path / "flux_model"
    (model_dir / "text_encoder").mkdir(parents=True)
    (model_dir / "text_encoder_2").mkdir(parents=True)
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "vae").mkdir(parents=True)

    (model_dir / "text_encoder" / "config.json").write_text(
        json.dumps({
            "architectures": ["CLIPTextModel"],
            "hidden_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 3,
            "intermediate_size": 24,
            "vocab_size": 99,
            "max_position_embeddings": 77,
        })
    )
    (model_dir / "text_encoder_2" / "config.json").write_text(
        json.dumps({
            "d_model": 64,
            "num_heads": 8,
            "d_kv": 8,
            "d_ff": 128,
            "num_layers": 3,
            "vocab_size": 321,
        })
    )

    def load_t5_weights(path, **kwargs):
        calls.setdefault("load_t5_weights", []).append((path, kwargs))
        return {"t5": np.array([1], dtype=np.float32)}

    def build_t5_encoder_engine(weights, **kwargs):
        calls.setdefault("build_t5_encoder_engine", []).append((weights, kwargs))
        return b"t5-plan"

    def load_clip_weights(path, **kwargs):
        calls["load_clip_weights"] = (path, kwargs)
        return {"clip": np.array([2], dtype=np.float32)}

    def build_clip_encoder_engine(weights, **kwargs):
        calls["build_clip_encoder_engine"] = (weights, kwargs)
        return b"clip-plan"

    def load_flux_dit_weights(path, **kwargs):
        calls["load_flux_dit_weights"] = (path, kwargs)
        return {"dit": np.array([3], dtype=np.float32)}

    def build_flux_dit_engine(weights, **kwargs):
        calls["build_flux_dit_engine"] = (weights, kwargs)
        return b"dit-plan"

    def build_flux_vae_decoder_engine(path, **kwargs):
        calls["build_flux_vae_decoder_engine"] = (path, kwargs)
        return b"vae-plan"

    monkeypatch.setitem(
        sys.modules,
        "trtf_build.t5_encoder_builder",
        _module(
            "trtf_build.t5_encoder_builder",
            load_t5_weights=load_t5_weights,
            build_t5_encoder_engine=build_t5_encoder_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "trtf_build.clip_encoder_builder",
        _module(
            "trtf_build.clip_encoder_builder",
            load_clip_weights=load_clip_weights,
            build_clip_encoder_engine=build_clip_encoder_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "trtf_build.flux_dit_builder",
        _module(
            "trtf_build.flux_dit_builder",
            load_flux_dit_weights=load_flux_dit_weights,
            build_flux_dit_engine=build_flux_dit_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "trtf_build.flux_vae_builder",
        _module(
            "trtf_build.flux_vae_builder",
            build_flux_vae_decoder_engine=build_flux_vae_decoder_engine,
        ),
    )

    monkeypatch.setattr(
        flux_mod,
        "_serialize_flux_preprocessor",
        lambda dit_weights, guidance_embeds: (
            calls.setdefault("serialize", []).append((dit_weights, guidance_embeds))
            or b"flux-preproc"
        ),
    )

    weights = {
        "_text_encoder_dir": str(model_dir / "text_encoder"),
        "_text_encoder_2_dir": str(model_dir / "text_encoder_2"),
        "_transformer_dir": str(model_dir / "transformer"),
        "_vae_dir": str(model_dir / "vae"),
        "_transformer_config": {
            "num_attention_heads": 2,
            "attention_head_dim": 4,
            "num_layers": 3,
            "num_single_layers": 4,
            "guidance_embeds": True,
        },
    }

    out = flux_mod.plugin.build_components(
        str(model_dir),
        _cfg(image_height=80, image_width=96),
        weights,
        verbose=False,
    )

    assert out["text_encoders"] == [("clip", b"clip-plan"), ("t5", b"t5-plan")]
    assert out["denoiser"] == b"dit-plan"
    assert out["vae_decoder"] == b"vae-plan"
    assert out["preprocessor_weights"] == b"flux-preproc"

    # h_lat=10, w_lat=12, pack_size=2 -> num_img_tokens=30
    assert calls["build_flux_dit_engine"][1]["num_img_tokens"] == 30
    assert calls["load_flux_dit_weights"][1]["dim"] == 8
    assert calls["serialize"][0][1] is True


def test_build_components_treats_text_encoder_as_t5_when_not_clip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Intent: cover branch where text_encoder directory contains a T5 config.

    Preconditions: text_encoder has non-CLIP architecture and no text_encoder_2 dir.
    Postconditions: T5 loader path is used and CLIP loader path is not used.
    """
    calls: dict[str, int] = {"clip_load": 0, "t5_load": 0}

    model_dir = tmp_path / "flux_model2"
    (model_dir / "text_encoder").mkdir(parents=True)
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "vae").mkdir(parents=True)

    (model_dir / "text_encoder" / "config.json").write_text(
        json.dumps({
            "architectures": ["T5EncoderModel"],
            "d_model": 48,
            "num_heads": 6,
            "d_kv": 8,
            "d_ff": 96,
            "num_layers": 2,
            "vocab_size": 123,
        })
    )

    def load_t5_weights(_path, **_kwargs):
        calls["t5_load"] += 1
        return {"t5": np.array([1], dtype=np.float32)}

    def build_t5_encoder_engine(_weights, **_kwargs):
        return b"t5-plan"

    def load_clip_weights(*_args, **_kwargs):
        calls["clip_load"] += 1
        return {"clip": np.array([1], dtype=np.float32)}

    monkeypatch.setitem(
        sys.modules,
        "trtf_build.t5_encoder_builder",
        _module(
            "trtf_build.t5_encoder_builder",
            load_t5_weights=load_t5_weights,
            build_t5_encoder_engine=build_t5_encoder_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "trtf_build.clip_encoder_builder",
        _module(
            "trtf_build.clip_encoder_builder",
            load_clip_weights=load_clip_weights,
            build_clip_encoder_engine=lambda *_a, **_k: b"clip-plan",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "trtf_build.flux_dit_builder",
        _module(
            "trtf_build.flux_dit_builder",
            load_flux_dit_weights=lambda *_a, **_k: {"dit": np.array([2], dtype=np.float32)},
            build_flux_dit_engine=lambda *_a, **_k: b"dit-plan",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "trtf_build.flux_vae_builder",
        _module(
            "trtf_build.flux_vae_builder",
            build_flux_vae_decoder_engine=lambda *_a, **_k: b"vae-plan",
        ),
    )
    monkeypatch.setattr(flux_mod, "_serialize_flux_preprocessor", lambda *_a, **_k: b"pre")

    weights = {
        "_text_encoder_dir": str(model_dir / "text_encoder"),
        "_transformer_dir": str(model_dir / "transformer"),
        "_vae_dir": str(model_dir / "vae"),
        "_transformer_config": {},
    }

    out = flux_mod.plugin.build_components(str(model_dir), _cfg(), weights, verbose=False)

    assert out["text_encoders"] == [("t5", b"t5-plan")]
    assert calls["t5_load"] == 1
    assert calls["clip_load"] == 0


def test_get_diffusion_config_guidance_toggle() -> None:
    """Intent: verify guidance-dependent scheduler defaults in diffusion config.

    Preconditions: config.raw contains transformer settings with/without guidance_embeds.
    Postconditions: step count and guidance scale follow the flag.
    """
    cfg_guided = _cfg(
        _transformer_config={
            "guidance_embeds": True,
            "num_attention_heads": 5,
            "attention_head_dim": 6,
        }
    )
    guided = flux_mod.plugin.get_diffusion_config(cfg_guided)
    assert guided["num_inference_steps"] == 28
    assert guided["guidance_scale"] == 3.5
    assert guided["guidance_embeds"] == 1
    assert guided["dit_dim"] == 30

    cfg_fast = _cfg(_transformer_config={"guidance_embeds": False})
    fast = flux_mod.plugin.get_diffusion_config(cfg_fast)
    assert fast["num_inference_steps"] == 4
    assert fast["guidance_scale"] == 0.0
    assert fast["guidance_embeds"] == 0


def test_serialize_flux_preprocessor_guidance_key_control() -> None:
    """Intent: validate key mapping and guidance key gating in serialized preprocessor blob.

    Preconditions: source dict includes base and guidance tensors.
    Postconditions: guidance keys are only serialized when guidance_embeds=True.
    """
    base = {
        "x_embedder.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "x_embedder.bias": np.array([1.0, 2.0], dtype=np.float32),
        "context_embedder.weight": np.arange(8, dtype=np.float32).reshape(2, 4),
        "context_embedder.bias": np.array([3.0, 4.0], dtype=np.float32),
        "time_text_embed.timestep_embedder.linear_1.weight": np.arange(4, dtype=np.float32).reshape(2, 2),
        "time_text_embed.timestep_embedder.linear_1.bias": np.array([5.0, 6.0], dtype=np.float32),
        "time_text_embed.timestep_embedder.linear_2.weight": np.arange(4, dtype=np.float32).reshape(2, 2),
        "time_text_embed.timestep_embedder.linear_2.bias": np.array([7.0, 8.0], dtype=np.float32),
        "time_text_embed.text_embedder.linear_1.weight": np.arange(4, dtype=np.float32).reshape(2, 2),
        "time_text_embed.text_embedder.linear_1.bias": np.array([9.0, 10.0], dtype=np.float32),
        "time_text_embed.text_embedder.linear_2.weight": np.arange(4, dtype=np.float32).reshape(2, 2),
        "time_text_embed.text_embedder.linear_2.bias": np.array([11.0, 12.0], dtype=np.float32),
        "time_text_embed.guidance_embedder.linear_1.weight": np.arange(4, dtype=np.float32).reshape(2, 2),
        "time_text_embed.guidance_embedder.linear_1.bias": np.array([13.0, 14.0], dtype=np.float32),
        "time_text_embed.guidance_embedder.linear_2.weight": np.arange(4, dtype=np.float32).reshape(2, 2),
        "time_text_embed.guidance_embedder.linear_2.bias": np.array([15.0, 16.0], dtype=np.float32),
    }

    idx_off, _payload_off = _decode_blob(flux_mod._serialize_flux_preprocessor(base, guidance_embeds=False))
    assert "condition_embedder.guidance_embedding.0.weight" not in idx_off

    idx_on, payload_on = _decode_blob(flux_mod._serialize_flux_preprocessor(base, guidance_embeds=True))
    assert "condition_embedder.guidance_embedding.0.weight" in idx_on

    max_end = 0
    for info in idx_on.values():
        nbytes = int(np.prod(info["shape"])) * 4
        max_end = max(max_end, info["offset"] + nbytes)
    assert len(payload_on) == max_end
