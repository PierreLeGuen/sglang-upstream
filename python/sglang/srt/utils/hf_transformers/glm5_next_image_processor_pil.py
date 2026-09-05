# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GLM-5 image preprocessing backport for the pinned Transformers runtime.

Source: huggingface/transformers@f62dc9bf2c90353b442a56e74391fbb8c689b55e
src/transformers/models/glm5_next/image_processing_pil_glm5_next.py

Only imports, documentation decorators, and the GLM-4V-compatible base/size
are adapted. Canvas alignment, padding, normalization and patch layout retain
upstream semantics; remove this fallback once the runtime ships GLM-5 Next.
"""

import math

import numpy as np
from transformers import Glm4vImageProcessorPil
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from transformers.processing_utils import Unpack
from transformers.utils import TensorType

from .glm5_next_image_processor import Glm5NextImageProcessorKwargs, smart_resize


class Glm5NextImageProcessorPil(Glm4vImageProcessorPil):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = (
        Glm4vImageProcessorPil.size
    )  # Required by the compatibility base; _preprocess uses token limits.
    default_to_square = False
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    valid_kwargs = Glm5NextImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]
    patch_expand_factor = 1
    min_image_tokens = 16
    max_image_tokens = 8000

    def preprocess(
        self, images: ImageInput, **kwargs: Unpack[Glm5NextImageProcessorKwargs]
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def resize(
        self,
        image: np.ndarray,
        resample: "PILImageResampling | int | None",
        factor: int,
        temporal_factor: int,
        min_image_tokens: int,
        max_image_tokens: int,
        **kwargs,
    ) -> np.ndarray:
        """Resize dynamically based on input image aspect ratio."""

        height, width = image.shape[-2:]
        target_height, target_width = smart_resize(
            height=height,
            width=width,
            num_frames=temporal_factor,
            factor=factor,
            temporal_factor=temporal_factor,
            min_pixels=min_image_tokens,
            max_pixels=max_image_tokens,
        )

        # Dynamic padded to ensure aspect ratio is compatible with `_patchify`
        pixels_per_token = temporal_factor * factor**2
        scale = min(target_height / height, target_width / width)
        if temporal_factor * height * width >= (pixels_per_token * min_image_tokens):
            scale = min(1.0, scale)
        content_height = max(1, min(target_height, math.floor(height * scale)))
        content_width = max(1, min(target_width, math.floor(width * scale)))

        # TODO: Also likely refactorable after min/max pixels has been added to size dict
        if (content_height, content_width) != (height, width):
            image = super().resize(
                image,
                SizeDict(height=content_height, width=content_width),
                resample=resample,
            )

        return np.pad(
            image,
            (
                (0, 0),
                (0, target_height - content_height),
                (0, target_width - content_width),
            ),
            mode="constant",
        )

    def patchify(
        self,
        image: np.ndarray,
        patch_size: int,
        merge_size: int,
        temporal_patch_size: int,
    ) -> tuple[np.ndarray, int, int]:
        """Patchifies each image into flat layout of shape (`seq_len`, `patch_dim`) so we can concat dynamically shaped pixels."""
        # Ensure float32 for patch processing
        image = np.asarray(image, dtype=np.float32)
        channel, resized_height, resized_width = image.shape
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

        patches = image.reshape(
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        # (gh, gw, mh, mw, C, ph, pw)
        patches = np.transpose(patches, (1, 4, 2, 5, 0, 3, 6))

        # expand temporal_patch_size as a broadcast (zero-copy)
        patches = np.broadcast_to(
            patches[:, :, :, :, :, None, :, :],
            (*patches.shape[:5], temporal_patch_size, *patches.shape[5:]),
        )

        flatten_patches = patches.reshape(
            grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )
        return flatten_patches, grid_h, grid_w

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_expand_factor: int,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        min_image_tokens: int,
        max_image_tokens: int,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess images one by one for PIL backend.
        """
        processed_images = []
        processed_grids = []

        for image in images:
            if do_resize:
                image = self.resize(
                    image,
                    resample=resample,
                    factor=patch_size * merge_size,
                    temporal_factor=temporal_patch_size,
                    min_image_tokens=min_image_tokens,
                    max_image_tokens=max_image_tokens,
                )

            # Rescale and normalize
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)

            patches, grid_h, grid_w = self.patchify(
                image,
                patch_size=patch_size,
                merge_size=merge_size,
                temporal_patch_size=temporal_patch_size,
            )

            # Remove batch dimension and append: shape is (seq_len, hidden_dim)
            processed_images.append(patches)
            processed_grids.append([1, grid_h, grid_w])

        # Concatenate all images along sequence dimension: (total_seq_len, hidden_dim)
        pixel_values = np.concatenate(processed_images, axis=0)
        image_grid_thw = np.array(processed_grids)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )

    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs: dict | None = None
    ) -> int:
        """
        A utility that returns number of image patches for a given image size.

        Args:
            height (`int`):
                Height of the input image.
            width (`int`):
                Width of the input image.
            images_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the image processor.
        Returns:
            `int`: Number of image patches per image.
        """
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)

        # Key difference is the dynamically based resize on min/max image tokens
        min_image_tokens = images_kwargs.get("min_image_tokens", self.min_image_tokens)
        max_image_tokens = images_kwargs.get("max_image_tokens", self.max_image_tokens)

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            num_frames=self.temporal_patch_size,
            height=height,
            width=width,
            factor=factor,
            min_pixels=min_image_tokens,
            max_pixels=max_image_tokens,
            temporal_factor=self.temporal_patch_size,
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w


__all__ = ["Glm5NextImageProcessorPil"]
