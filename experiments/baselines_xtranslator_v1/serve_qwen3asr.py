"""X-Translator's own Qwen3-ASR adapter (server/qwen3asr/qwen3asr_server.py, commit dc6a59a) on a chosen port.

The adapter hard-codes port 8001 in main(), so several instances cannot run side by side. This launcher imports the
adapter module unchanged and repeats its main() (same Qwen3ASRModel.LLM arguments: max_new_tokens 32,
trust_remote_code), changing only the port and the resource flags. --chunk-size-sec (registered point xt-a1.6 only)
replaces the adapter's literal streaming chunk 0.6 s in init_streaming_state calls; default 0.6 = unchanged.
"""
import argparse
import sys

XT = "/opt/dlami/nvme/jasonleeeli/research_sources/baselines_20261112/X-Translator/server/qwen3asr"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--chunk-size-sec", type=float, default=0.6)
    ap.add_argument("--kv-cache-memory-bytes", default=None, help="fixed KV cache size (resource only), e.g. 1536M")
    a = ap.parse_args()
    sys.path.insert(0, XT)
    import uvicorn
    import qwen3asr_server as S
    from qwen_asr import Qwen3ASRModel

    extra = {}
    if a.kv_cache_memory_bytes:
        from vllm.engine.arg_utils import human_readable_int
        extra["kv_cache_memory_bytes"] = human_readable_int(a.kv_cache_memory_bytes)
    S.model = Qwen3ASRModel.LLM(model=a.model, gpu_memory_utilization=a.gpu_memory_utilization,
                                max_model_len=a.max_model_len, max_new_tokens=32, trust_remote_code=True, **extra)
    if a.chunk_size_sec != 0.6:
        orig = S.model.init_streaming_state

        def init_streaming_state(*args, **kw):
            if kw.get("chunk_size_sec") == 0.6:
                kw["chunk_size_sec"] = a.chunk_size_sec
            return orig(*args, **kw)

        S.model.init_streaming_state = init_streaming_state
    print(f"[serve_qwen3asr] port={a.port} chunk_size_sec={a.chunk_size_sec}", flush=True)
    uvicorn.run(S.app, host="127.0.0.1", port=a.port, workers=1)


if __name__ == "__main__":
    main()
