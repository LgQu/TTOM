"""
CogVideoX-specific TTOM guidance wrapper.

Provides a high-level interface for running TTOM-style test-time optimization
on CogVideoX latents using cross-attention layout guidance.
"""
import torch
from typing import Dict, List, Optional, Any
from dsl.guidance.guidance_pipeline import LatentGuidance
from dsl.utils.layout_handler import interpolate_layout_boxes


class CogVideoXGuidance:
    """
    High-level wrapper for TTOM guidance on CogVideoX models.
    Handles layout interpolation, guidance parameter setup, and optimization loop.
    """

    def __init__(
        self,
        scheduler,
        transformer,
        seed: int = 42,
    ):
        self.latent_guidance = LatentGuidance(
            scheduler=scheduler,
            transformer=transformer,
            seed=seed,
        )

    def prepare(
        self,
        layout: dict,
        num_latent_frames: int,
        guidance_attn_strategy: str = "middle_2",
        max_iter: int = 5,
        lr: float = 5e-3,
        loss_threshold: float = 12.0,
        loss_scale: float = 1.0,
        set_amf_loss: bool = True,
        set_latents_norm: bool = False,
    ) -> dict:
        """
        Prepare guidance parameters from a layout dictionary.

        Returns:
            backward_guidance_kwargs dict ready to pass to the pipeline.
        """
        bboxes = interpolate_layout_boxes(layout, num_interp_frames=num_latent_frames)

        kwargs = {
            "bboxes": bboxes,
            "loss_scale": loss_scale,
            "loss_threshold": loss_threshold,
            "set_amf_loss": set_amf_loss,
            "max_iter": max_iter,
            "lr": lr,
            "verbose": True,
            "clear_cache": True,
            "guidance_attn_strategy": guidance_attn_strategy,
            "guidance_attn_keys": {},
            "set_latents_norm": set_latents_norm,
        }

        self.latent_guidance.prepare_parameters(**kwargs)
        return kwargs

    def step(
        self,
        latents: torch.Tensor,
        cond_embeddings: torch.Tensor,
        timestep: torch.Tensor,
        image_rotary_emb: Optional[torch.Tensor],
        attention_kwargs: Optional[Dict[str, Any]],
        object_positions_id_dict: Dict[str, List[int]],
        latent_frames: int,
        step_index: int,
        energy_function_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Run a single guidance optimization step.
        """
        loss = torch.tensor(10000.0)

        result = self.latent_guidance(
            cond_embeddings=cond_embeddings,
            attention_kwargs=attention_kwargs,
            index=step_index,
            t=timestep,
            image_rotary_emb=image_rotary_emb,
            latents=latents,
            loss=loss,
            model_type="cogvideox",
            object_positions_id_dict=object_positions_id_dict,
            latent_frames=latent_frames,
            energy_function_kwargs=energy_function_kwargs or {},
        )

        return result[0], result[1]
