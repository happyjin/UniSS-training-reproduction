#!/usr/bin/env bash
# change 47+: p3<TAG>_report.sh from p3d3_report.sh (drafter <TAG>/merged, <TAG>/h/params.json, <TAG>/dev/choice.json, output p3<TAG>).
#   bash experiments/sst_hsmm_v1/make_report.sh <TAG>     (refuses to overwrite)
set -eu
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS/experiments/sst_hsmm_v1
T=$1; O=p3${T}_report.sh
[ -e $O ] && { echo "exists $O" >&2; exit 1; }
{ echo "#!/usr/bin/env bash"
  echo "# change 47+: test p3$T = p3d3_report.sh with drafter $T/merged, $T/h/params.json, $T/dev/choice.json (made by make_report.sh)."
  tail -n +2 p3d3_report.sh | sed -e '/^#/!s#sst_hsmm_v1/d3/#sst_hsmm_v1/'$T'/#g' -e '/^#/!s#p3d3#p3'$T'#g'; } > $O
grep -n "$T" $O | grep -v "^[0-9]*:#" 
