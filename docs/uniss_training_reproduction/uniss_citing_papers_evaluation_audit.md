# 引用 UniSS 论文的外部评测审计

> 审计日期：2026-07-26  
> 对应本地报告：`eval_outputs/uniss_full198_phase2_phase3_20260726T065924Z/report/dev_only/phase2_phase3_dev_detailed_analysis.md` 第10.2节

## 核心结论

- 没有引用论文报告 UniST `dev-00000.parquet` 或 `test-00000.parquet` 上的 UniSS 数值。
- STEB 和 OpenSTBench 重新运行了官方 UniSS，但测试集分别是 STEB 和 OpenSTBench 的多数据集协议。
- COMPASS、CoT-TTS Challenge、X-Translator 只引用 UniSS 的指标实践、三阶段范式或相关工作，没有运行 UniSS checkpoint。

## 逐篇审计

| 论文 | 是否运行 UniSS | UniSS规模/来源 | 数据 | 文献方法与模型规模 | 与本地0.5B的差异 |
| --- | --- | --- | --- | --- | --- |
| STEB [1] | 是 | 论文未重述参数和mode；按UniSS原文/官方模型归类为1.5B级 | STEB 32.6小时 ZH↔EN | benchmark；Qwen3-30B-A3B仅作为judge | 不同数据、ASR和expressiveness指标；官方模型约为本地3倍参数 |
| OpenSTBench [2] | 是 | 官方Hugging Face UniSS，1.5B，官方VAD分块 | MSLT dev 1,000条/方向，另有LibriTTS、RAVDESS、MCAE-SPPS等 | 评测框架无单一参数量；对照SeamlessM4T-v2-Large为2.3B | 不同数据和指标；OpenSTBench BLEU是文本侧，CER/WER不是Speech-BLEU |
| COMPASS [3] | 否 | - | FLEURS/CVSS，1,248个模型-语言配置 | 46指标、8维度评测框架，无单一参数量 | 只可借鉴评测工具，不能提供UniSS性能基线 |
| ISCSLP 2026 CoT-TTS [4] | 否 | - | CoT-TTS训练集与600中文+600英文评测集 | 0.6B Qwen3-based、参数高效微调、三阶段训练 | 参数接近但任务是上下文TTS，不是跨语言S2ST |
| X-Translator [5] | 否 | - | OpenSTBench、长语音、多说话人、FLEURS | 1.7B ASR + 8B MT + 0.4B TTS，名义约10.1B的分离式cascade | 实时cascade与单阶段0.5B模型的延迟、显存和参数含义不同 |

## 外部实测数值与本地结果

下表不计算差值或排名，因为测试集和指标实现不同。

| 来源 | 模型 | 数据/方向 | Text-BLEU | 最终语音BLEU | UTMOS | SLC-0.2 | SLC-0.4 | Speaker | Emotion | RTF |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 本地 Phase3 Performance | 0.5B | UniST dev EN→ZH | 37.8105 | 35.9432 | 3.0876 | 0.7225 | 0.9365 | - | - | - |
| 本地 Phase3 Quality | 0.5B | UniST dev EN→ZH | 44.8822 | 42.8128 | 3.1044 | 0.7197 | 0.9344 | - | - | - |
| 本地 Phase3 Performance | 0.5B | UniST dev ZH→EN | 33.3860 | 1.4670 | 3.4991 | 0.5953 | 0.8366 | - | - | - |
| 本地 Phase3 Quality | 0.5B | UniST dev ZH→EN | 40.4631 | 1.8927 | 3.5021 | 0.5960 | 0.8326 | - | - | - |
| STEB [1] | 官方/原文1.5B级，mode未披露 | STEB EN→ZH | 48.91 | 46.43 | - | 0.980 | 0.990 | 0.291 WavLM-SIM | 3.82/5 LLM judge | - |
| STEB [1] | 官方/原文1.5B级，mode未披露 | STEB ZH→EN | 30.87 | 28.55 | - | 0.915 | 0.959 | 0.411 WavLM-SIM | 3.61/5 LLM judge | - |
| OpenSTBench [2] | 官方1.5B | MSLT等，EN→ZH | 34.1008 | - | 3.2409 | 0.9940 | 0.9980 | 0.8468 Resemblyzer | 0.7383 E2V | 1.5449 |
| OpenSTBench [2] | 官方1.5B | MSLT等，ZH→EN | 18.7520 | - | 3.4139 | 0.9919 | 0.9960 | 0.8459 Resemblyzer | 0.9035 E2V | 1.0838 |

## 口径解释

- STEB最终语音BLEU使用Qwen3-ASR；本地使用中文Paraformer、英文Whisper-large-v3。
- OpenSTBench BLEU来自文本侧translation-quality表；CER/WER评估生成语音相对中间文本的实现误差。
- OpenSTBench的speaker和emotion结果使用专门的LibriTTS/RAVDESS/MCAE-SPPS子集，不是全部来自MSLT dev。
- STEB Emotion是1--5分LLM judge，OpenSTBench E2V是embedding cosine，二者不可横向比较。
- 本地0.5B约为官方1.5B参数量的三分之一；训练数据、checkpoint、VAD/裁剪、推理mode和ASR backend也不同。
- X-Translator的约10.1B是三个串联生成模块的参数和，不等价于一个10.1B联合模型。

## 参考文献

1. Sitong Cheng et al. **STEB: A Speech-to-Speech Translation Expressiveness Benchmark for Evaluating Beyond Translation Fidelity.** arXiv:2606.25529, 2026. <https://arxiv.org/abs/2606.25529>
2. Yanjie An et al. **OpenSTBench: Beyond Semantic Evaluation for Speech Translation.** arXiv:2605.30792, 2026. <https://arxiv.org/abs/2605.30792>
3. Alkis Koudounas et al. **Benchmarking Speech-to-Speech Translation Models.** arXiv:2606.03241, 2026. <https://arxiv.org/abs/2606.03241>
4. Wei Xue et al. **ISCSLP 2026 CoT-TTS Challenge: Chain-of-Thought Reasoning for Context-Aware Text-to-Speech.** arXiv:2606.21933, 2026. <https://arxiv.org/abs/2606.21933>
5. Yuxiang Zhao et al. **X-Translator: A Real-Time Multilingual Speaker-Aware Speech-to-Speech Translation System.** arXiv:2607.17544, 2026. <https://arxiv.org/abs/2607.17544>
6. Sitong Cheng et al. **UniSS: Unified Expressive Speech-to-Speech Translation with Your Voice.** arXiv:2509.21144, 2025. <https://arxiv.org/abs/2509.21144>
