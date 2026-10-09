"""Load Qwen2.5-Omni's processor without pulling in the vision branch.

The stock ``Qwen2_5OmniProcessor`` declares four sub-processors, and one of
them -- ``Qwen2VLVideoProcessor`` -- refuses to instantiate without
torchvision. This box does not have torchvision and must not get it: the
conda env is a fixed Megatron/CUDA stack that in-flight training and
evaluation runs depend on.

S2TT feeds audio and text only, so neither vision branch does any work.
``replace_multimodal_special_tokens`` nonetheless reads ``merge_size`` off
both processors at the top of the function, before the per-token dispatch
that would actually use them. The image processor instantiates fine
(torchvision is a video-only dependency) so it is kept as-is; the video
processor is replaced by a shim carrying that one geometry field, taken
from the checkpoint's own config. The value is never consumed -- only the
``video_token`` branch reads it, and that branch cannot fire without
videos -- but reading it must not raise.

Everything the parent's ``__init__`` does after the super() call is read
token ids off the tokenizer, which we reproduce verbatim below.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from transformers import (
    AutoTokenizer,
    Qwen2_5OmniProcessor,
    Qwen2VLImageProcessor,
    WhisperFeatureExtractor,
)
from transformers.processing_utils import ProcessorMixin


@dataclass(frozen=True)
class _VideoGeometryShim:
    """Stands in for the video processor's geometry fields.

    Only ``merge_size`` is ever read on the audio+text path, and only
    eagerly -- never to compute anything.
    """

    merge_size: int


class AudioTextOmniProcessor(Qwen2_5OmniProcessor):
    """``Qwen2_5OmniProcessor`` with the video branch removed."""

    attributes = ["image_processor", "feature_extractor", "tokenizer"]

    def __init__(
        self,
        image_processor=None,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        video_geometry=None,
    ):
        ProcessorMixin.__init__(
            self, image_processor, feature_extractor, tokenizer,
            chat_template=chat_template,
        )
        self.video_processor = video_geometry
        self.image_token = self.tokenizer.image_token
        self.audio_token = self.tokenizer.audio_token
        self.video_token = self.tokenizer.video_token
        self.vision_bos_token = self.tokenizer.vision_bos_token
        self.vision_eos_token = self.tokenizer.vision_eos_token
        self.audio_bos_token = self.tokenizer.audio_bos_token
        self.audio_eos_token = self.tokenizer.audio_eos_token


def load_audio_text_processor(model_dir: str | Path) -> AudioTextOmniProcessor:
    """Build the audio+text processor from the checkpoint's own components."""
    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    image_processor = Qwen2VLImageProcessor.from_pretrained(str(model_dir))
    feature_extractor = WhisperFeatureExtractor.from_pretrained(str(model_dir))
    chat_template = json.loads(
        (model_dir / "chat_template.json").read_text(encoding="utf-8")
    )["chat_template"]
    merge_size = json.loads(
        (model_dir / "preprocessor_config.json").read_text(encoding="utf-8")
    )["merge_size"]
    return AudioTextOmniProcessor(
        video_geometry=_VideoGeometryShim(merge_size=merge_size),
        image_processor=image_processor,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        chat_template=chat_template,
    )
