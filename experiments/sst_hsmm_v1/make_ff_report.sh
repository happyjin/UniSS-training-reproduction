#!/usr/bin/env bash
# change 48: p3<TAG>_report.sh from p3d3_report.sh for faithful finalisation: Thinker = wave17 + hsc17 --draft-lora <DL> (non-final
# reads), HSMM params <PRM>, spec from <TAG>/dev/choice.json, output p3<TAG>.   bash make_ff_report.sh <TAG> <DL> <PRM>  (refuses to overwrite)
set -eu
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS/experiments/sst_hsmm_v1
T=$1; DL=$2; PRM=$3; O=p3${T}_report.sh
[ -e $O ] && { echo "exists $O" >&2; exit 1; }
{ echo "#!/usr/bin/env bash"
  echo "# change 48: test p3$T = p3d3_report.sh with wave17 + hsc17 --draft-lora $DL (non-final reads), params $PRM, spec $T/dev/choice.json (make_ff_report.sh)."
  tail -n +2 p3d3_report.sh | sed -e '/^#/!s#^NTH=eval_outputs/sst_hsmm_v1/d3/merged#NTH=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05#' \
    -e '/^#/!s#--params eval_outputs/sst_hsmm_v1/d3/h/params.json#--params '$PRM' --draft-lora '$DL'#' \
    -e '/^#/!s#hsc16_si_thinker#hsc17_si_thinker#g' -e '/^#/!s#sst_hsmm_v1/d3/dev/#sst_hsmm_v1/'$T'/dev/#g' -e '/^#/!s#p3d3#p3'$T'#g'; } > $O
grep -c "hsc17_si_thinker.*--draft-lora $DL" $O
