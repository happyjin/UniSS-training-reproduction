#!/usr/bin/env python3
# =============================================================================
# training-metrics-report / parse_metrics.py
# =============================================================================
# 把 fetch_logs.sh 拉下来的 raw_log.txt（HF Trainer logging）+ mfu_log.jsonl
# 解析成一份结构化 JSON report，供写入 Markdown 多列对照文档使用。
# 纯本地解析，不发起任何网络 / kubectl 调用。
#
# 用法：
#   python3 parse_metrics.py --raw-log <raw_log.txt> [--mfu-log <mfu_log.jsonl>]
#           [--steps 1,10,20,...,200] [--steady-window 191-200]
#
# 输出：JSON 写到 stdout，结构见本文件末尾 report 字典的注释。
# =============================================================================
import argparse
import ast
import json
import re
import sys


TRAINER_LINE_RE = re.compile(r"\{'loss':.*?'train_speed\(s/it\)':\s*[0-9.eE+-]+\}")

# deepspeed 链路专属字段 vs megatron 链路专属字段（用于自动判断后端，仅供参考展示，
# 不影响解析逻辑——两套字段都会原样保留在输出里）。
DEEPSPEED_ONLY_FIELDS = {"token_text_acc", "token_acoustic_token"}
MEGATRON_ONLY_FIELDS = {"codec_loss", "codec_token_acc", "codec_residual_loss"}


def parse_trainer_log(raw_log_path):
    """从 raw_log.txt 提取所有 HF Trainer logging 字典行，按 step 去重（保留最后一次）。"""
    text = open(raw_log_path, "r", errors="ignore").read()
    records = {}
    backend = "unknown"
    for m in TRAINER_LINE_RE.finditer(text):
        line = m.group(0)
        try:
            d = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
        step = None
        if "global_step/max_steps" in d:
            step = int(d["global_step/max_steps"].split("/")[0])
            backend = "deepspeed"
        elif "iteration" in d:
            step = int(d["iteration"].split("/")[0])
            backend = "megatron"
        if step is None:
            continue
        d["_step"] = step
        records[step] = d  # 同 step 覆盖，天然保留"最后一次出现"
    return backend, dict(sorted(records.items()))


def parse_mfu_log(mfu_log_path):
    """解析 mfu_log.jsonl，自动截取最后一段（检测 step==1 重新出现的位置）。"""
    lines = [l.strip() for l in open(mfu_log_path, "r", errors="ignore") if l.strip()]
    all_records = []
    for l in lines:
        if not l.startswith("{"):
            continue
        try:
            all_records.append(json.loads(l))
        except json.JSONDecodeError:
            continue
    if not all_records:
        return []
    # 找最后一次 step==1 出现的下标，从那里开始才是"本次运行"的数据
    # （同一个 OUTPUT_DIR 被多次提交会导致文件里出现多段历史）
    last_reset_idx = 0
    for i, r in enumerate(all_records):
        if r.get("step") == 1:
            last_reset_idx = i
    return all_records[last_reset_idx:]


def fmt_duration(seconds):
    seconds = int(round(seconds))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if d:
        parts.append(f"{d}d")
    if d or h:
        parts.append(f"{h}h")
    parts.append(f"{m}m")
    parts.append(f"{s}s")
    return "".join(parts)


def summarize_mfu(records, steady_start, steady_end):
    by_step = {r["step"]: r for r in records}
    steady = [by_step[s] for s in range(steady_start, steady_end + 1) if s in by_step]
    overall = [by_step[s] for s in range(2, steady_end + 1) if s in by_step]
    warmup = by_step.get(1)

    def avg(lst, key):
        vals = [x[key] for x in lst]
        return sum(vals) / len(vals) if vals else None

    def mn(lst, key):
        vals = [x[key] for x in lst]
        return min(vals) if vals else None

    def mx(lst, key):
        vals = [x[key] for x in lst]
        return max(vals) if vals else None

    cum_sec = sum(by_step[s]["step_sec"] for s in range(1, steady_end + 1) if s in by_step)
    missing_steps = [s for s in range(1, steady_end + 1) if s not in by_step]

    return {
        "steady_window": f"{steady_start}-{steady_end}",
        "steady": {
            "step_sec_avg": avg(steady, "step_sec"), "step_sec_min": mn(steady, "step_sec"), "step_sec_max": mx(steady, "step_sec"),
            "mfu_avg": avg(steady, "mfu"), "mfu_min": mn(steady, "mfu"), "mfu_max": mx(steady, "mfu"),
            "tflops_avg": avg(steady, "tflops"),
            "tokens_per_sec_avg": avg(steady, "tokens_per_sec"),
        },
        "overall_excl_warmup": {
            "step_sec_avg": avg(overall, "step_sec"),
            "mfu_avg": avg(overall, "mfu"),
            "tflops_avg": avg(overall, "tflops"),
            "tokens_per_sec_avg": avg(overall, "tokens_per_sec"),
        },
        "warmup_step1": warmup,
        "cumulative_step_sec_sum": cum_sec,
        "cumulative_duration_fmt": fmt_duration(cum_sec) if cum_sec else None,
        "missing_steps": missing_steps,
        "record_count": len(records),
        "step_range": [records[0]["step"], records[-1]["step"]] if records else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-log", required=True)
    ap.add_argument("--mfu-log", default=None)
    ap.add_argument("--steps", default="1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200")
    ap.add_argument("--steady-window", default="191-200")
    args = ap.parse_args()

    steps = [int(s) for s in args.steps.split(",")]
    steady_start, steady_end = (int(x) for x in args.steady_window.split("-"))

    backend, trainer_records = parse_trainer_log(args.raw_log)

    trainer_at_steps = {}
    for s in steps:
        d = trainer_records.get(s)
        trainer_at_steps[s] = d if d else None

    report = {
        "backend": backend,
        "trainer_log": {
            "total_records": len(trainer_records),
            "step_range": [min(trainer_records), max(trainer_records)] if trainer_records else None,
            "at_steps": trainer_at_steps,
        },
        "mfu_log": None,
    }

    if args.mfu_log:
        try:
            mfu_records = parse_mfu_log(args.mfu_log)
            report["mfu_log"] = summarize_mfu(mfu_records, steady_start, steady_end) if mfu_records else None
        except FileNotFoundError:
            report["mfu_log"] = None

    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main()
