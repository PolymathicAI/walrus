from dataclasses import replace
from functools import reduce
from operator import mul
from typing import Callable, List, Optional

import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from the_well.data.datasets import BoundaryCondition

from walrus.models.shared_utils.flexi_utils import (
    choose_kernel_size_deterministic,
    choose_kernel_size_random,
)
from walrus.models.shared_utils.normalization import RMSGroupNorm
from walrus.models.shared_utils.patch_jitterers import (
    FixedPatchJittererBoundaryPad,
    PatchJittererBoundaryPad,
)


def dim_pad(x, max_d):
    """
    Assume T B C are first channels, then see how many spatial dims we need to append/
    """
    squeeze = 0
    if x.ndim - 3 < max_d:
        x = x.unsqueeze(-1)
        squeeze += 1
    if x.ndim - 3 < max_d:
        x = x.unsqueeze(-1)
        squeeze += 1
    return x, squeeze


class IsotropicModel(nn.Module):
    """
    Naive model that operates at a single dimension with a repeating block.

    Args:
        patch_size (tuple): Size of the input patch
        hidden_dim (int): Dimension of the embedding
        processor_blocks (int): Number of blocks (consisting of spatial mixing - temporal attention)
        n_states (int): Number of input state variables.
    """

    def __init__(
        self,
        encoder,
        decoder,
        processor,
        projection_dim: int = 96,
        intermediate_dim: int = 192,
        hidden_dim: int = 768,
        processor_blocks: int = 8,
        n_states: int = 4,
        drop_path: float = 0.2,
        input_field_drop: float = 0.1,
        groups: int = 12,
        max_d: int = 3,
        jitter_patches: bool = True,
        use_periodic_fixed_jitter: bool = False,
        gradient_checkpointing_freq: int = 0,
        causal_in_time: bool = False,
        include_d: List[int] = [2, 3],  # Temporary due to FSDP resume issue
        override_dimensionality: Optional[
            int
        ] = 0,  # Temporary due to FSDP resume issue
        dim_key_override: Optional[int] = None,  # Temporary due to FSDP resume issue
        norm_layer: Callable = RMSGroupNorm,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.drop_path = drop_path
        self.max_d = max_d
        self.jitter_patches = jitter_patches
        self.dp = np.linspace(0, drop_path, processor_blocks)
        self.causal_in_time = causal_in_time
        self.gradient_checkpointing_freq = gradient_checkpointing_freq
        self.override_dimensionality = override_dimensionality
        self.dim_key_override = dim_key_override
        self.encoder_dummy = nn.Parameter(
            torch.ones(1)
        )  # for grad checkpointing, see: https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/11
        self.hidden_dim = hidden_dim
        if (
            self.override_dimensionality is not None
            and self.override_dimensionality > 0
        ):
            include_d = [self.override_dimensionality]
        self.input_field_drop = input_field_drop
        if use_periodic_fixed_jitter:
            self.patch_jitterer = FixedPatchJittererBoundaryPad(
                stage_dim=projection_dim,
                patch_size=None,
                max_d=self.max_d,
                jitter_patches=jitter_patches,
            )
        else:
            self.patch_jitterer = PatchJittererBoundaryPad(
                stage_dim=projection_dim,
                patch_size=None,
                max_d=self.max_d,
                jitter_patches=jitter_patches,
            )
        self.embed = nn.ModuleDict(
            {
                str(i): encoder(
                    spatial_dims=3,
                    input_dim=n_states,
                    inner_dim=intermediate_dim,
                    output_dim=hidden_dim,
                    groups=groups,
                    norm_layer=norm_layer,
                )
                for i in range(1, self.max_d + 1)
                if i in include_d
            }
        )

        self.blocks = nn.ModuleList(
            [
                processor(
                    hidden_dim=hidden_dim,
                    drop_path=self.dp[i],
                    causal_in_time=causal_in_time,
                    gradient_checkpointing=(
                        i % gradient_checkpointing_freq == 0
                        if gradient_checkpointing_freq > 0
                        else False
                    ),
                    norm_layer=norm_layer,
                )
                for i in range(processor_blocks)
            ]
        )
        self.debed = nn.ModuleDict(
            {
                str(i): decoder(
                    input_dim=hidden_dim,
                    inner_dim=intermediate_dim,
                    output_dim=n_states,
                    spatial_dims=3,
                    groups=groups,
                    norm_layer=norm_layer,
                )
                for i in range(1, self.max_d + 1)
                if i in include_d
            }
        )

    def add_ft_options(
        self,
        ft_param_dict: dict,
    ):
        """
        Add options for fine-tuning the model.
        """
        # APE
        if "ape_shape" in ft_param_dict:
            print("activated ape")
            shape = ft_param_dict["ape_shape"]
            self.ape = nn.Parameter(5e-3 * torch.randn(1, 1, self.hidden_dim, *shape))

        if "learnable_rope" in ft_param_dict and ft_param_dict["learnable_rope"]:
            print("activated learnable rope", ft_param_dict.get("rope_per_axis", False))
            for blk in self.blocks:
                if hasattr(blk, "make_rope_learnable"):
                    blk.make_rope_learnable(
                        per_axis=ft_param_dict.get("rope_per_axis", False)
                    )

        if "freeze" in ft_param_dict:
            raise NotImplementedError("Freezing not implemented yet")

    def _encoder_forward(
        self,
        x,
        state_labels,
        bcs,
        metadata,
        patch_size,
        dynamic_ks=None,
        encoder_dummy=None,
    ):
        if self.override_dimensionality > 0:
            n_spatial_dims = metadata.n_spatial_dims
        else:
            n_spatial_dims = sum([int(dim != 1) for dim in x.shape[3:]])
        if self.dim_key_override is None:
            dim_key = str(n_spatial_dims)
        else:
            dim_key = str(self.dim_key_override)
        T, B = x.shape[:2]
        # Project into higher dim
        x = rearrange(
            x, "t b c h ... -> b c (t h) ..."
        )  # Field dropout is intended to drop out the entire field. We could either implement our own mask or reshape to use existing function and this was slightly faster
        x = F.dropout3d(
            x, training=self.training, p=self.input_field_drop / x.shape[1]
        )  # Bonferonni correction for variable fields - all
        x = rearrange(x, "b c (t h) ... -> t b c h ...", t=T)
        x = (
            x * encoder_dummy
        )  # NOTE - this is just a single scalar to work around a bug in PyTorch's grad checkpointing - if this moves away from zero, we can add it to the space bag weights in postprocessing
        # x = self.space_bag(x, state_labels)
        # x = rearrange(x, "t b ... c -> t b c ...")
        # Now encoder
        if (
            hasattr(self.embed[dim_key], "learned_pad")
            and self.embed[dim_key].learned_pad
        ):
            x, jitter_info = self.patch_jitterer(
                x,
                bcs[0],
                metadata,
                patch_size=patch_size,
                learned_pad=self.embed[dim_key].learned_pad,
                random_kernel=dynamic_ks,
                base_kernel=self.embed[dim_key].base_kernel_size,
            )
        else:
            x, jitter_info = self.patch_jitterer(
                x, bcs[0], metadata, patch_size=patch_size
            )

        # Sparse proj
        state_labels = torch.cat(
            [
                state_labels,
                torch.tensor(
                    [2, 0, 1], device=state_labels.device, dtype=state_labels.dtype
                ),
            ],
            dim=0,
        )
        x, stage_info = self.embed[dim_key](
            x, state_labels, bcs[0], metadata, random_kernel=dynamic_ks
        )
        if hasattr(self, "ape"):
            x = x + self.ape
        return x, stage_info, jitter_info

    def _decoder_forward(self, x, state_labels, bcs, stage_info, jitter_info, metadata):
        """Run the decoder and invert the jitter"""
        if self.override_dimensionality > 0:
            n_spatial_dims = metadata.n_spatial_dims
        else:
            n_spatial_dims = sum([int(dim != 1) for dim in x.shape[3:]])
        if self.dim_key_override is None:
            dim_key = str(n_spatial_dims)
        else:
            dim_key = str(self.dim_key_override)
        x = self.debed[dim_key](x, state_labels, bcs[0], stage_info, metadata)
        if (
            hasattr(self.embed[dim_key], "learned_pad")
            and self.embed[dim_key].learned_pad
        ):
            x = self.patch_jitterer.unjitter(
                x, jitter_info, learned_pad=self.embed[dim_key].learned_pad
            )
        else:
            x = self.patch_jitterer.unjitter(x, jitter_info)
        return x

    def forward(
        self,
        x,
        state_labels,
        bcs,
        metadata,
        proj_axes=None,
        return_att=False,
        train=True,
    ):
        # x - T B C H [W D]
        # state_labels - C
        # bcs - #dims, 2
        # proj axes - #dims - Permutes axes to discourage learning axes - dependent relationships
        # NOTE: Everything gets padded to max_d below, so we want the metadata to reflect this
        metadata = replace(metadata, n_spatial_dims=self.max_d)
        n_spatial_dims = metadata.n_spatial_dims
        dim_key = str(n_spatial_dims)
        # Pad to max dims so we can just use 3D convs - same flops, but empirically would be faster
        # to dynamically adjust which conv is used, but more verbose for compiler-friendly version
        x, squeeze_out = dim_pad(x, self.max_d)
        T, B, C = x.shape[:3]
        x_shape = x.shape[3:]

        dynamic_ks = []
        patch_size = []

        if metadata.dataset_name == "post_neutron_star_merger":
            # Neutron star has a size 66 dimension - just interpolate to 64 for simplicity
            x = rearrange(x, "t b c h w d -> (t b) c h w d")
            x = F.interpolate(
                x,
                size=(x.shape[2], x.shape[3], 64),
                mode="trilinear",
                align_corners=False,
            )
            x = rearrange(x, "(t b) c h w d -> t b c h w d", t=T)
            x_shape = x.shape[3:]
        # Choose the variable patches if applicable
        if (
            hasattr(self.embed[dim_key], "variable_downsample")
            and (self.embed[dim_key].variable_downsample)
            and self.embed[dim_key].variable_deterministic_ds
        ):
            # support for variable but deterministic downsampling
            dynamic_ks = choose_kernel_size_deterministic(x_shape)
            patch_size = [reduce(mul, k) for k in dynamic_ks]
            # patch_size doesn't matter for the dimension that is higher than the number of spatial dims
            patch_size.extend([0] * (self.max_d - len(patch_size)))

        # support for variable and random downsampling.
        # this will probably not be used in Walrus but a needed feature for dedicated paper
        elif hasattr(self.embed[dim_key], "variable_downsample") and (
            self.embed[dim_key].variable_downsample
        ):
            for _ in range(self.max_d):
                ks = (
                    choose_kernel_size_random(self.embed[dim_key].kernel_scales_seq)
                    if train
                    else (2, 2)
                )
                patch_size.append(ks[0] * ks[1])
                dynamic_ks.append(ks)
            dynamic_ks = tuple(dynamic_ks)
        # constant downsampling as with hmlp
        else:
            patch_size = [self.embed[dim_key].patch_size] * self.max_d
        # Always assume we need to checkpoint the encoder if any checkpointing is on
        if self.gradient_checkpointing_freq > 0:
            x, stage_info, jitter_info = torch.utils.checkpoint.checkpoint(
                self._encoder_forward,
                x,
                state_labels,
                bcs,
                metadata,
                patch_size,
                dynamic_ks,
                self.encoder_dummy,
                use_reentrant=False,
            )
        else:
            x, stage_info, jitter_info = self._encoder_forward(
                x,
                state_labels,
                bcs,
                metadata,
                patch_size,
                dynamic_ks,
                self.encoder_dummy,
            )

        # Process
        all_att_maps = []
        # Compute a periodic roll
        # Blk inputs are T, B, C, H, W, D
        periodic_dims = []
        for dim in range(len(bcs[0])):
            if bcs[0][dim][0] == BoundaryCondition["PERIODIC"].value:
                periodic_dims.append(dim + 3)
        periodic_dim_shapes = [x.shape[dim] for dim in periodic_dims]
        roll_total = [0] * len(periodic_dims)
        for ii, blk in enumerate(self.blocks):
            # Randomly roll dimensions of x corresponding to periodic BCs
            if len(periodic_dims) > 0 and self.jitter_patches:
                roll_quantities = [
                    np.random.randint(0, periodic_dim_shapes[dim])
                    for dim in range(len(periodic_dims))
                ]
                roll_total = [
                    roll_quantities[dim] + r for dim, r in enumerate(roll_total)
                ]
                x = torch.roll(
                    x,
                    shifts=roll_quantities,
                    dims=periodic_dims,
                )
            x, att_maps = blk(x, bcs, return_att=return_att)
            all_att_maps += att_maps
        # If we randomly rolled, we need to roll back
        if sum(roll_total) > 0 and self.jitter_patches:
            x = torch.roll(
                x,
                shifts=[-r for r in roll_total],
                dims=periodic_dims,
            )
        # Decode
        # If not causal, no need to debed all time steps so just take the last one
        if not self.causal_in_time:
            x = x[-1:]

        if self.gradient_checkpointing_freq > 0:
            x = torch.utils.checkpoint.checkpoint(
                self._decoder_forward,
                x,
                state_labels,
                bcs,
                stage_info,
                jitter_info,
                metadata,
                use_reentrant=False,
            )
        else:
            x = self._decoder_forward(
                x, state_labels, bcs, stage_info, jitter_info, metadata
            )

        # If neutron star, interpolate back to original size
        if metadata.dataset_name == "post_neutron_star_merger":
            T = x.shape[0]
            x = rearrange(x, "t b c h w d -> (t b) c h w d")
            x = F.interpolate(
                x,
                size=(x.shape[2], x.shape[3], 66),
                mode="trilinear",
                align_corners=False,
            )
            x = rearrange(x, "(t b) c h w d -> t b c h w d", t=T)

        # De-inflate the extra channels if they were added:
        for _ in range(squeeze_out):
            x = x.squeeze(-1)
        # Return T, B, C, H, [W], [D]
        return x  # TODO - Return attention maps for debugging

    def compile(self, **kwargs) -> None:
        warnings.warn("IsotropicModel.compile() currently only supports torch compilation for IsotropicModel.blocks. Note that torch.compile(IsotropicModel) will attempt to compile the entire forward call instead", UserWarning)
        for blk in self.blocks:
            blk.compile(**kwargs)