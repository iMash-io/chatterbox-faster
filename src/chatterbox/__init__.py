try:
    from importlib.metadata import version, PackageNotFoundError  # Python 3.8+
except ImportError:
    from importlib_metadata import version, PackageNotFoundError  # For Python <3.8

try:
    __version__ = version("chatterbox-tts")
except PackageNotFoundError:
    # Running from source without installed package metadata
    __version__ = "0.0.0"

from .tts import ChatterboxTTS
from .vc import ChatterboxVC
from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
