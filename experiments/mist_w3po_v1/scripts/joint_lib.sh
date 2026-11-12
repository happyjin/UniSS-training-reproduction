# Shared definitions of the joint bidirectional run (b) launchers (joint_stage1.sh, joint_stage2.sh, joint_proxy.sh,
# joint_dev.sh, joint_test.sh; reports/mist_w3po_v1/joint_bidir_plan_prereg_v1.zh-CN.md). Sourced, never run.
J=/opt/dlami/nvme/jasonleeeli
export PYTHONPATH=.:third_party/Megatron-LM TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface \
  MODELSCOPE_CACHE=$J/cache/modelscope OMP_NUM_THREADS=8
PY=$J/conda_envs/qwen3omni/bin/python
SC=experiments/mist_w3po_v1/scripts
JL=logs/mist_w3po_v1/joint                      # launcher logs, orchestrator run dirs, marker files
JO=eval_outputs/mist_w3po_v1/joint              # checkpoints converted / merged, selections
SL=logs/mist_w3po_v1/slots
DEC=eval_outputs/mist_w3po_v1/decode
SPD=eval_outputs/mist_w3po_v1/speech
TE=eval_outputs/mist_w3po_v1/crf/t_eval
CRF=eval_outputs/mist_w3po_v1/crf
DD=eval_outputs/mist_w3po_v1/data
O2=eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_o2_20261029b/thinker_merged
V8Z=eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dl_zh2en_3k_20261030/thinker_merged
V8E=eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dl_20261029b/thinker_merged
APRIME_CK=$CRF/train/crf_v2p_20261110T075131Z/ckpt/step_03000
E2Z=logs/mist_w3po_v1/en2zh                     # experiment (a)
# windows / references
WZ_DEV=$DD/windows_zh2en_c2g06_v1/windows_dev_online.jsonl
WZ_DEV5=$DD/windows_zh2en_c2g06_v1_dev5/windows_dev_online.jsonl
WZ_DEVC=eval_outputs/mist_w3po_v1/detector_v1/windows_chunk_v1/windows_dev_online.jsonl
WZ_DEV5C=$JO/windows_zh2en_chunk_dev5/windows_dev_online.jsonl
WE_DEV=$DD/windows_en2zh_c2g06_v1/windows_dev_online.jsonl
WE_DEV5=$DD/windows_en2zh_c2g06_v1_dev5/windows_dev_online.jsonl
WE_DEVC=$DD/windows_en2zh_c2g06_chunk_v1/windows_dev_online.jsonl
WE_DEV5C=$DD/windows_en2zh_c2g06_chunk_v1_dev5/windows_dev_online.jsonl
RE_DEV=data/ami_multispk_si_v1/e2e_gstar_v1/windows_validation_ref.jsonl
RE_TEST=data/ami_multispk_si_v1/e2e_gstar_v1/windows_test_ref.jsonl
WT_AMI=$DD/windows_test_c2g06_v1/windows_test_online_zh2en.jsonl
WT_ALI=$DD/windows_test_c2g06_v1/windows_test_online_ali.jsonl
WT_EN=$DD/windows_test_en2zh_c2g06_v1/windows_test_online_en2zh.jsonl
WT_AMIC=$DD/windows_test_c2g06_chunk_v1/zh2en/windows_dev_online.jsonl
WT_ALIC=$DD/windows_test_c2g06_chunk_v1/ali/windows_dev_online.jsonl
WT_ENC=$DD/windows_test_en2zh_c2g06_chunk_v1/windows_dev_online.jsonl
RT_AMI=data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_ref.jsonl
RT_ALI=data/diarist_alimeeting_refs_v1/windows_test_ref_human.jsonl
JORCH="$PY $SC/joint_orch.py"
TORCHZ="$PY $SC/thinker_t_eval_orch.py"
EORCH="$PY $SC/en2zh_orch.py"
GIB=1024   # MiB per GiB
CAP_MIB=$((135 * GIB))
log() { echo "$(date -u +%FT%TZ) $*"; }
wait_done() { until [ -e $1/events/DONE ]; do sleep 30; done; }
orch_failed() { tail -3 $1/orch.log 2>/dev/null | grep -q "DONE; failed/skipped \[\]" && echo "" || echo "failed"; }
is_baseline_eval() {   # <pid>: the process or an ancestor runs a single-speaker paper baseline evaluation
  local p=$1 cmd i
  for i in $(seq 1 12); do
    [ -z "$p" ] || [ "$p" -le 1 ] && return 1
    cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null) || return 1
    case "$cmd" in *s2st_baselines_v1*|*baselines_xtranslator_v1*|*baselines_hysimulmt*|*baselines_results*|*/baselines_*) return 0;; esac
    p=$(ps -o ppid= -p $p 2>/dev/null | tr -d ' ')
  done
  return 1
}
others_on() {   # <cards "4,5,6,7"> -> lines "<card> <pid> <MiB> <placeholder|baseline|other>" for every compute process
  local cards=$1 map pid bus mem idx cmd kind
  map=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>/dev/null) || { echo "x 0 0 smi-failed"; return; }
  nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv,noheader,nounits 2>/dev/null | \
  while IFS=', ' read -r pid bus mem; do
    idx=$(echo "$map" | awk -F', ' -v b="$bus" '$2 == b {print $1}')
    case ",$cards," in *",$idx,"*) ;; *) continue;; esac
    cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null) || cmd="(unreadable)"
    case "$cmd" in
      *scripts/gpu_load/train_mist_workload.py*|*gpu_keeper.py*) kind=placeholder;;
      *) if is_baseline_eval $pid; then kind=baseline; else kind=other; fi;;
    esac
    echo "$idx $pid $mem $kind"
  done
}
cards_ready() {   # <cards> <training peak GiB>: empty output = only placeholders, and baseline evals within 135 GiB
  local cards=$1 peak=$2
  others_on $cards | awk -v cap=$CAP_MIB -v peak=$((peak * GIB)) '
    $4 == "other" || $4 == "smi-failed" {print "busy " $1 ":" $2; next}
    $4 == "baseline" {use[$1] += $3}
    END {for (c in use) if (use[c] + peak > cap) print "baseline-over-budget card " c ": " use[c] " MiB + peak " peak}'
}
merge_ok() {   # <decode dir of the merged copy> <windows file>: complete output and pair_lost <= 2 -> prints json, rc 0/1
  $PY - "$1" "$2" <<'EOF'
import json, sys
m = json.load(open(f"{sys.argv[1]}/merge_read_segments.json"))
n = sum(1 for _ in open(sys.argv[2]))
ok = m["windows"] == n and m["pair_lost"] <= 2
print(json.dumps({"windows": m["windows"], "expected": n, "pair_lost": m["pair_lost"], "go": ok}))
sys.exit(0 if ok else 1)
EOF
}
read_merge() {   # <src decode name> <sep> <windows file> <log>: merge_read_segments -> <src>_rl; rc 0 = go
  # pair_lost <= 2 with a complete copy: go (the registered rule). More: go only under the coordinator's 2026-11-12 22:20
  # pre-authorisation pattern (every lost window the known mechanism, pair_lost <= 1 % of the windows, copy complete),
  # logged as JOINT_MERGE_DEVIATION; otherwise rc 1 (the caller logs JOINT_STOP).
  local src=$1 sep=$2 win=$3 lg=$4 n pl okm hl
  [ -e $DEC/${src}_rl/merge_read_segments.json ] || \
    $PY -m experiments.mist_w3po_v1.crf.merge_read_segments --src $src --dst ${src}_rl --sep "$sep" >> $lg 2>&1
  merge_ok $DEC/${src}_rl $win >> $lg 2>&1 && return 0
  [ -e $DEC/${src}_rl/merge_read_segments.json ] || return 1
  n=$(wc -l < $win)
  [ -e $JO/pairlost_$src.json ] || CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.merge_pairlost_check \
    --src $src > $JO/pairlost_$src.json 2>> $lg || return 1
  read pl okm <<< "$($PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['pair_lost'], int(d['all_known_mechanism']))" $JO/pairlost_$src.json)" || return 1
  hl=$(wc -l < $DEC/${src}_rl/hyp.jsonl 2>/dev/null || echo 0)
  if [ "$okm" = 1 ] && [ $((pl * 100)) -le $n ] && [ "$hl" = $n ] && [ -s $DEC/${src}_rl/score_text.json ]; then
    log "JOINT_MERGE_DEVIATION $src: pair_lost $pl of $n windows, all the known mechanism, copy complete (pre-authorised pattern)"
    return 0
  fi
  return 1
}
en2zh_failed() {   # experiment (a)'s stop markers
  grep -hE "EN2ZH_STOP|EN2ZH_PROXY_FAILED" $E2Z/ckpt_proxy.log $E2Z/after_select.log $E2Z/test.log 2>/dev/null | tail -1
}
en2zh_talker() {   # (a)'s selected PL-EM export dir (EN2ZH_TALKER_SELECTED iteration N (<dir>)), empty until then
  sed -n 's/.*EN2ZH_TALKER_SELECTED iteration [0-9]* (\(.*\))$/\1/p' $E2Z/after_select.log 2>/dev/null | tail -1
}
a_thinker_exited() {   # (a)'s Thinker training on 0-3 has exited (CRF_RUN_EXIT), or (a) stopped: 0-3 are no training cards
  local r; r=$(cat $E2Z/RUN_ID 2>/dev/null)
  [ -n "$r" ] && grep -q CRF_RUN_EXIT logs/mist_w3po_v1/$r.log 2>/dev/null && return 0
  [ -n "$(en2zh_failed)" ]
}
