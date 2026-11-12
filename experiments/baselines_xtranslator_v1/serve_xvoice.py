"""X-Translator's own X-Voice adapter (server/xvoice/server.py, commit dc6a59a), run as __main__ with the arguments of
server/xvoice/start_x_voice.sh. Environment shims only (no change to synthesis):
  * fasttext under numpy 2 (see below);
  * torchaudio 2.9 routes torchaudio.load through torchcodec / FFmpeg, which this machine lacks: load wav files with
    soundfile instead (same float32 [channels, frames] tensor, PCM16 scaled by 1/32768 as torchaudio does);
  * DeepFilterNet 0.5.6 (X-Voice's reference denoiser) imports torchaudio.backend.common.AudioMetaData, removed from
    torchaudio; a stub class makes the import succeed (enhance() does not use it).
"""
import runpy
import sys
import types

import numpy as np
import soundfile as sf
import torch
import torchaudio

XT_SERVER = "/opt/dlami/nvme/jasonleeeli/research_sources/baselines_20261112/X-Translator/server/xvoice/server.py"


def _sf_load(uri, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None, buffer_size=4096,
             backend=None):
    x, sr = sf.read(uri, dtype="float32", always_2d=True, start=frame_offset,
                    frames=num_frames if num_frames and num_frames > 0 else -1)
    t = torch.from_numpy(np.ascontiguousarray(x.T if channels_first else x))
    return t, sr


torchaudio.load = _sf_load

# fasttext 0.9.3 (used by X-Voice's fastlid language detection) calls np.array(..., copy=False), which numpy 2 rejects;
# X-Voice then silently falls back to the requested language. np.asarray is what numpy < 2 did there.
import fasttext.FastText as _ft  # noqa: E402


class _NumpyCopyCompat:
    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def array(obj, *args, copy=True, **kw):
        if copy is False:
            return np.asarray(obj, *args, **kw)
        return np.array(obj, *args, copy=copy, **kw)


_ft.np = _NumpyCopyCompat()
if "torchaudio.backend.common" not in sys.modules:
    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")

    class AudioMetaData:  # noqa: D401 - import stub only
        pass

    common.AudioMetaData = AudioMetaData
    backend.common = common
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common

sys.argv = [XT_SERVER] + sys.argv[1:]
runpy.run_path(XT_SERVER, run_name="__main__")
