# UniSS Offline--Streaming S2ST 中文技术论文

本目录独立保存 Offline Phase 1--3、Streaming Stage A/B 和拟议 RL/GRPO 阶段的中文 LaTeX 技术论文，不修改历史训练、评估、checkpoint 或 demo 资产。

## 文件

- `main.tex`：论文正文。
- `references.bib`：参考文献。
- `figures/architecture.tikz.tex`：可复现 TikZ 系统图。
- `build/main.pdf`：编译后的 PDF。
- `build/main.log`：编译日志。

## 编译

编译器安装在用户要求的本地根目录下：

```text
/opt/dlami/nvme/jasonleeeli/tools/tectonic-0.15.0/tectonic
```

执行：

```bash
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS/docs/uniss_streaming_s2st_technical_paper_v1
make
```

论文严格区分已完成实验与未来 RL 方案；Stage A/B 的失败 gate、Stage A teacher-KL denominator 为零、LAAL/ATD proxy 等边界均在正文中显式说明。
