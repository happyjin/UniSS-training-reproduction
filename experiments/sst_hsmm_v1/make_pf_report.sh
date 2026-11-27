#!/usr/bin/env bash
# portfolio 2026-11-27 (change 55): test script p3<TAG>_report.sh = p3d3_report.sh on the card pool with <THINKER>, Thinker module
# <ONLINE> (e.g. experiments.sst_hsmm_v1.hsc22_si_thinker), params <PRM>, extra flags "<XF>" and the spec of <TAG>/dev/choice.json.
#   bash make_pf_report.sh <TAG> <ONLINE> <PRM> "<XF>" <THINKER>      (refuses to overwrite)
set -eu
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS/experiments/sst_hsmm_v1
T=$1; ON=$2; PRM=$3; XF=$4; TH=$5; O=p3${T}_report.sh
[ -e $O ] && { echo "exists $O" >&2; exit 1; }
{ echo "#!/usr/bin/env bash"
  echo "# portfolio change 55: test p3$T = p3d3_report.sh (card pool) with $TH, $ON $XF, params $PRM, spec $T/dev/choice.json (make_pf_report.sh)."
  tail -n +2 p3d3_report.sh | sed -e '/^#/!s#^NTH=eval_outputs/sst_hsmm_v1/d3/merged#NTH='$TH'#' \
    -e '/^#/!s#--params eval_outputs/sst_hsmm_v1/d3/h/params.json#--params '$PRM' '"$XF"'#' \
    -e '/^#/!s#experiments\.sst_hsmm_v1\.hsc16_si_thinker#'$ON'#g' -e '/^#/!s#sst_hsmm_v1/d3/dev/#sst_hsmm_v1/'$T'/dev/#g' -e '/^#/!s#p3d3#p3'$T'#g' \
    -e 's#^source experiments/sst_hsmm_v1/lib.sh$#source experiments/sst_hsmm_v1/lib.sh\nsource experiments/sst_hsmm_v1/lib_pool.sh#'; } > $O
grep -c "$ON .*--params $PRM" $O >/dev/null && grep -c "lib_pool.sh" $O
