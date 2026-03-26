import os
from typing import Any, Dict, List, Optional
import torch
from dsl.guidance import energy_functions
from dsl.guidance.amf_loss import amf_loss
import gc
import numpy as np
import warnings
import random


class LatentGuidance:
    def __init__(
        self, 
        scheduler, 
        transformer,
        seed
    ):
        self.scheduler = scheduler
        self.transformer = transformer
        self.seed = seed

    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def prepare_parameters(
        self,
        **backward_guidance_kwargs,
    ):
        """
        Method to prepare and validate parameters.
        This function ensures all necessary parameters are available and properly formatted.
        If a parameter is missing, a warning will be issued.
        """
        
        def get_param(param_name, default_value):
            """Helper function to fetch parameter with warning if not provided."""
            if param_name not in backward_guidance_kwargs:
                if param_name == "guidance_attn_keys":
                    warnings.warn(f"Parameter '{param_name}' not provided. Using strategy: {self.guidance_attn_strategy}", UserWarning)
                else:
                    warnings.warn(f"Parameter '{param_name}' not provided. Using default: {default_value}", UserWarning)
            return backward_guidance_kwargs.get(param_name, default_value)

        self.bboxes = get_param("bboxes", None)
        self.loss_scale = get_param("loss_scale", 0.5)
        self.loss_threshold = get_param("loss_threshold", 0.2)
        self.max_iter = get_param("max_iter", 5)
        self.lr = get_param("lr", 5e-3)
        self.verbose = get_param("verbose", False)
        self.clear_cache = get_param("clear_cache", False)
        self.set_amf_loss = get_param("set_amf_loss", False)
        self.guidance_attn_strategy = get_param("guidance_attn_strategy", "all")
        self.set_guidance_attn_keys(set(get_param("guidance_attn_keys", {})))
        self.set_latents_norm = get_param("set_latents_norm", False)

    def set_guidance_attn_keys(self, guidance_attn_keys: set=None):
        num_layers=self.transformer.config.num_layers
        print("\nnum_blocks of transformer: ", num_layers)
        if self.guidance_attn_strategy == "all":
            self.guidance_attn_keys = {i for i in range(0, num_layers)}
        elif self.guidance_attn_strategy == "first":
            self.guidance_attn_keys = {0}
        elif self.guidance_attn_strategy == "last":
            self.guidance_attn_keys = {self.max_iter - 1}
        elif self.guidance_attn_strategy.startswith("interval"):
            _, interval = self.guidance_attn_strategy.split("_")
            self.guidance_attn_keys = {i for i in range(0, num_layers, int(interval))}
        elif self.guidance_attn_strategy.startswith("middle"):
            _, n = self.guidance_attn_strategy.split("_")
            n = int(n)
            if n > num_layers:
                raise ValueError(f"Requested {n} middle layers, but model only has {num_layers} layers.")
            start = (num_layers - n) // 2
            self.guidance_attn_keys = {i for i in range(start, start + n)}
        elif self.guidance_attn_strategy.startswith("custom"):
            self.guidance_attn_keys = guidance_attn_keys
            print("guidance_attn_keys: ", self.guidance_attn_keys)
        else:
            raise ValueError(f"Invalid guidance_attn_strategy: {self.guidance_attn_strategy}")

    def get_attn_maps_cogvideox(
        self,
        latent_model_input,
        cond_embeddings,
        timestep,
        image_rotary_emb,
        attention_kwargs,
        object_positions_id_list,
    ):
        ret = self.transformer(
            hidden_states = latent_model_input,
            encoder_hidden_states=cond_embeddings,
            timestep=timestep,
            image_rotary_emb=image_rotary_emb,
            attention_kwargs=attention_kwargs,
            object_positions_id_list=object_positions_id_list,
            return_attn=True,
            return_dict=False,
            saved_attention_block_ids=self.guidance_attn_keys
        )
        return ret

    def get_attn_maps_wan(
        self,
        latent_model_input,
        timestep,
        prompt_embeds,
        attention_kwargs,
        object_positions_id_list,
    ):
        ret = self.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            attention_kwargs=attention_kwargs,
            object_positions_id_list=object_positions_id_list,
            return_attn=True,
            return_AMF=self.set_amf_loss,
            return_dict=False,
            saved_attention_block_ids=self.guidance_attn_keys
        )
        return ret


    def __call__(
        self,
        cond_embeddings: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        index: int = 0,
        t: torch.Tensor = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        latents: torch.Tensor = None,
        loss: torch.Tensor = None,
        model_type: str = "cogvideox",
        return_guidance_saved_attn: bool = False,
        object_positions_id_dict: Dict[str, List[int]] = None,
        latent_frames: int = None,
        energy_function_kwargs: Optional[Dict[str, Any]] = None,
    ):
        iteration = 0
        attention_probs_to_return_iter_0 = None
        attention_probs_to_return_iter_max = None
        latents_param = torch.nn.Parameter(latents.detach().clone(), requires_grad=True)
        
        ratio_final = 0.2
        total_steps = 10
        lr = self.lr * (1 - (1 - ratio_final) * index / (total_steps - 1))
        print(f"lr: {lr}")
        optimizer = torch.optim.Adam([latents_param], lr=lr)

        if isinstance(self.max_iter, list):
            self.max_iter = self.max_iter[index]

        with torch.set_grad_enabled(True):
            while (
                loss.item() > self.loss_threshold
                and iteration < self.max_iter
            ):
                self.set_seed(42)
                latent_model_input = latents_param
                # latents.requires_grad_(True)
                # latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                timestep = t.expand(latent_model_input.shape[0])
                if model_type == "cogvideox":
                    self.transformer._set_gradient_checkpointing(self.transformer, value=True)
                else:
                    self.transformer.gradient_checkpointing = True
                    print("--- gradient_checkpointing success ---")
                object_positions_id_list = sorted(set(token_id for obj_list in object_positions_id_dict.values() for token_id in obj_list))
                AMF_map = None

                if model_type == "wan":
                    out_dir = f"/hpctmp/e1351271/lvd/LVD_extention/out/attn_maps/wan/test"
                    # os.makedirs(out_dir, exist_ok=True)
                    ret = self.get_attn_maps_wan(
                        latent_model_input,
                        timestep,
                        prompt_embeds,
                        attention_kwargs,
                        object_positions_id_list,
                    )
                    attention_probs_all = ret["attention_probs_all"]
                    torch.save(attention_probs_all, os.path.join(out_dir, f"attention_probs_all_{index}.pt"))
                    if self.set_amf_loss:
                        AMF_map = ret["AMF_map"]                   
                        torch.save(AMF_map, os.path.join(out_dir, f"AMF_map_{index}.pt"))
                elif model_type == "cogvideox":
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    # out_dir = f"/hpctmp/e1351271/lvd/LVD_extention/out/attn_maps/cogvideo/test"
                    # os.makedirs(out_dir, exist_ok=True)
                    ret = self.get_attn_maps_cogvideox(
                        latent_model_input,
                        cond_embeddings,
                        timestep,
                        image_rotary_emb,
                        attention_kwargs,
                        object_positions_id_list,
                    )
                    attention_probs_all = ret["attention_probs_all"]
                    # torch.save(attention_probs_all, os.path.join(out_dir, f"attention_probs_all_{index}.pt"))
                    if self.set_amf_loss:
                        AMF_map = ret["AMF_maps_all"]                   
                        # torch.save(AMF_map, os.path.join(out_dir, f"AMF_map_{index}.pt"))                    
                else:
                    raise ValueError(f"Invalid model_type: {model_type}")                    

                # update latents with guidance
                if self.set_amf_loss:
                    loss = amf_loss(
                        saved_attn=attention_probs_all,
                        amf_map=AMF_map,
                        bboxes=self.bboxes,
                        object_positions_id_dict=object_positions_id_dict,
                        guidance_attn_keys=self.guidance_attn_keys,
                        index=index,
                        latent_frames=latent_frames,
                        verbose=self.verbose,
                    )
                else:
                    loss = (
                        energy_functions.compute_ca_lossv3(
                            saved_attn=attention_probs_all,
                            bboxes=self.bboxes,
                            object_positions_id_dict=object_positions_id_dict,
                            guidance_attn_keys=self.guidance_attn_keys,
                            index=index,
                            latent_frames=latent_frames,
                            verbose=self.verbose,
                            **energy_function_kwargs,
                        )
                        * self.loss_scale
                    )

                loss = loss.to(dtype=latents.dtype)
                print("loss: ", loss)
                if torch.isnan(loss):
                    print("**Loss is NaN**")
                for group in optimizer.param_groups:
                    print("Learning Rate:", group['lr'])

                # with torch.no_grad():
                    # latents_before_update = latents_param.clone()

                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_([latents_param], max_norm=1.0)
                optimizer.step()  

                # grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]
                # latents.requires_grad_(False)
                # lr = self.lr * t / 1000
                # latents = latents - lr * grad_cond

                del attention_probs_all
                if self.set_amf_loss:
                    del AMF_map
                gc.collect()
                torch.cuda.empty_cache()

                if self.set_latents_norm:
                    with torch.no_grad():
                        print("--- set_latents_norm ---")
                        B = latents_param.shape[0]
                        d = latents_param[0].numel()
                        expected_norm = d ** 0.5
                        latent_norm = latents_param.view(B, -1).norm(dim=1)
                        correction = (expected_norm / (latent_norm + 1e-8)).view(B, 1, 1, 1)
                        print(f"correction: {correction}")
                        latents_param.mul_(correction)

                if self.clear_cache:
                    gc.collect()
                    torch.cuda.empty_cache()

                if self.verbose:
                    print(
                        f"time index {index}, loss: {loss.item():.3f}, loss threshold: {self.loss_threshold:.3f}, iteration: {iteration}"
                    )
                    
                iteration += 1

        if return_guidance_saved_attn:
            return latents_param.detach(), loss, attention_probs_to_return_iter_0, attention_probs_to_return_iter_max
        else:
            return latents_param.detach(), loss