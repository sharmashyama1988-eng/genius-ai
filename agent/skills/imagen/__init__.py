"""Image Generation Package (Imagen / Custom API).

User-customizable image generation module supporting free zero-key generation
(Pollinations / HuggingFace) and custom user API keys (DALL-E, Stability, FLUX).
"""

from .image_generator import generate_image

__all__ = ["generate_image"]
