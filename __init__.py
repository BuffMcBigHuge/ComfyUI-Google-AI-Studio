"""
ComfyUI Google AI Studio Custom Nodes
Provides integration with Google AI Studio APIs including TTS, image generation, and more.
"""

from .google_ai_studio_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

WEB_DIRECTORY = "./web" 