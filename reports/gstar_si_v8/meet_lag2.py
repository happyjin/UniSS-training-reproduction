"""Stop (by PID) the AliMeeting Lag2 resume shard r3s<k> and its reverse helper r3h<k> once together they cover every
window of shard k of resume3.windows.jsonl; shards without a helper are left alone. Merging is by window id."""
import json, os, signal, subprocess, sys, time
os.chdir("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
R = "eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/runs/on_ali_S2dllag2"
W = [json.loads(l)["id"] for l in open(f"{R}/resume3.windows.jsonl")]
left = {int(k) for k in sys.argv[1:]}


def ids(f):
    out = set()
    if os.path.exists(f):
        for l in open(f):
            try:
                out.add(json.loads(l)["id"])
            except Exception:
                pass
    return out


while left:
    for k in sorted(left):
        if set(W[k::3]) <= ids(f"{R}/hyp_r3s{k}.jsonl") | ids(f"{R}/hyp_r3h{k}.jsonl"):
            for ln in subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines():
                pid, _, a = ln.strip().partition(" ")
                if a.startswith("/") and "gstar_si_stream_online_v6s" in a and \
                        (f"on_ali_S2dllag2/hyp_r3s{k}.jsonl" in a or f"on_ali_S2dllag2/hyp_r3h{k}.jsonl" in a):
                    os.kill(int(pid), signal.SIGTERM)
                    print(time.strftime("%H:%M:%S"), "covered shard", k, "stopped", pid, flush=True)
            left.discard(k)
    time.sleep(60)
print("MEET_DONE", flush=True)
