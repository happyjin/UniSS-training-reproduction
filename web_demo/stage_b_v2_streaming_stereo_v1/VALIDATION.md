# Validation record

## Reused environment

The recovered historical environment is reused without installing or
downloading packages:

```text
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-offline-demo
Python 3.12.13
Gradio 5.49.1
PyTorch 2.6.0+cu124
Torchaudio 2.6.0+cu124
Transformers 4.53.1
Librosa 0.11.0
FFmpeg / FFprobe 8.1.2
```

The launcher confines HOME, temporary files and Gradio, Hugging Face, Torch
and kernel caches under `/opt/dlami/nvme/jasonleeeli/`.

## CPU and audio-tool validation

Four `unittest` checks pass under the reused environment. They cover the
public UI contract, both stereo players, cached/full causal token parity and
the legacy finalizer compatibility contract. FFmpeg successfully generated a
16 kHz mono PCM WAV and FFprobe read it back as `pcm_s16le`, 16000 Hz, one
channel.

## Real H200 upload smoke

The fixed 5.46 s Chinese `magicdata_0000000001` source sample completed on one
H200:

```text
translation = I want to search for text messages on Baidu.Please.
first WRITE / first timeline audio = 5120 ms
fallback used = false
forced actions = 0
structural recoveries = 1
initial load plus inference = 40.46 s
translation/timeline/stereo WAV = non-empty
```

## Real H200 microphone-prefix smoke

The same waveform was delivered as browser-like 640 ms PCM increments. The
Student consumed each increment internally as cached 160 ms frames with 80 ms
right context:

```text
policy events = WAIT at 3200, 3840, 4480 and 5120 ms; WRITE at 5460 ms
translation = I want to use Baidu to search for text messages.
first WRITE / first timeline audio = 5460 ms
fallback used = false
source frontend RTF at final event = about 0.138
stereo = 2 channels, 16 kHz, non-empty
```

The source waveform contains leading silence. With a 20 ms RMS audible gate,
the left source channel first crossed the gate at 1.90 s and the right target
channel at 6.90 s, an audible channel offset of about 5.00 s. This shows that
the causal Student frontend is computationally fast, but the retained R2
policy still waited until the end of this sample. It is not evidence of
sub-second end-to-end speech output.
