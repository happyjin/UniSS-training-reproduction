# Unsafe forced-WRITE diagnostic

这些文件仅用于听取被安全质量门拦截的训练分布外语音，不能视为正常streaming结果。

| sample | forced text | audio seconds | coverage | failures |
|---|---|---:|---:|---|
| [train_en_zh_01](train_en_zh_01/unsafe_forced_streaming_translation.wav) | (empty) | 0.00 | 0.0% | no_safe_streaming_translation, no_natural_write, forced_write_dominant, audio_coverage:0.0000 |
| [train_en_zh_02](train_en_zh_02/unsafe_forced_streaming_translation.wav) | 这 | 0.24 | 3.8% | no_natural_write, forced_write_dominant, unsafe_forced_audio_emitted, audio_coverage:0.0380 |
| [train_zh_en_01](train_zh_en_01/unsafe_forced_streaming_translation.wav) | Well | 0.24 | 5.4% | no_natural_write, forced_write_dominant, unsafe_forced_audio_emitted, audio_coverage:0.0543 |
| [train_zh_en_02](train_zh_en_02/unsafe_forced_streaming_translation.wav) | I | 0.24 | 5.0% | no_natural_write, forced_write_dominant, unsafe_forced_audio_emitted, audio_coverage:0.0496 |
| [dev_en_zh_01](dev_en_zh_01/unsafe_forced_streaming_translation.wav) | 李 | 0.24 | 7.4% | no_natural_write, forced_write_dominant, unsafe_forced_audio_emitted, audio_coverage:0.0741 |
| [dev_en_zh_02](dev_en_zh_02/unsafe_forced_streaming_translation.wav) | (empty) | 0.00 | 0.0% | no_safe_streaming_translation, no_natural_write, forced_write_dominant, audio_coverage:0.0000 |
| [dev_zh_en_01](dev_zh_en_01/unsafe_forced_streaming_translation.wav) | I | 0.24 | 4.4% | no_natural_write, forced_write_dominant, unsafe_forced_audio_emitted, audio_coverage:0.0440 |
| [dev_zh_en_02](dev_zh_en_02/unsafe_forced_streaming_translation.wav) | (empty) | 0.00 | 0.0% | no_safe_streaming_translation, no_natural_write, forced_write_dominant, audio_coverage:0.0000 |
