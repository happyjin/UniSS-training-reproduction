"""Stop (by PID) the AliMeeting LA2 forward shard s<k> and its reverse helper h<k> once together they cover every window
of shard k (they decode from opposite ends); merging is by window id, so nothing is scored twice."""
import json, os, signal, subprocess, time
R = "eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/runs/on_ali_S2dlLA2"
W = [json.loads(l)["id"] for l in open("data/online_v6_20261031/windows_test_online_ali.jsonl")]
os.chdir("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
def ids(f):
    out = set()
    if os.path.exists(f):
        for l in open(f):
            try: out.add(json.loads(l)["id"])
            except Exception: pass
    return out
left = {0, 1}
while left:
    for k in sorted(left):
        if set(W[k::2]) <= ids(f"{R}/hyp_s{k}.jsonl") | ids(f"{R}/hyp_h{k}.jsonl"):
            ps = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines()
            for ln in ps:
                pid, _, a = ln.strip().partition(" ")
                if a.startswith("/") and "gstar_si_stream_online_v6s" in a and (f"on_ali_S2dlLA2/hyp_s{k}.jsonl" in a or f"on_ali_S2dlLA2/hyp_h{k}.jsonl" in a):
                    os.kill(int(pid), signal.SIGTERM)
                    print(time.strftime("%H:%M:%S"), "covered shard", k, "stopped", pid, flush=True)
            left.discard(k)
    time.sleep(60)
print("MEET_DONE", flush=True)
