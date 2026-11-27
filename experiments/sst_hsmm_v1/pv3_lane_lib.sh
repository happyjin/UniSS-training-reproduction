# plan v3 (change 56): method env for the work-stealing lanes (pf_lane_lib2.sh run). Sourced after pf_lane_lib2.sh.
P2FT="--draft-lora $D0/p2ft/peft/a1.0000 --draft-tag"
TRW() { until [ -d $D0/h1tr/train ] && [ -d $D0/h4tr/train ]; do sleep 30; done; }      # never take cards between smoke and train
E1D1() { run pfe1d1 ROLL=1 FIT=hsmm_anchor XF="$P2FT --mid-mbr 6 --mbr-util support"; }
H3() { run pfh3 WAIT=$D0/pv3/H3_SMOKE_OK ROLL=0 PRM=$D0/pfe1/h/params.json XF="$P2FT --syn-gate"; }
H1() { run pfh1 WAIT=$D0/h1tr/TRAIN_DONE ROLL=1 FIT=hsmm_anchor XF="--draft-lora $D0/h1tr/peft/a1.0000 --draft-tag --tag-set h1"; }
H4() { run pfh4 WAIT=$D0/h4tr/TRAIN_DONE ROLL=1 FIT=hsmm_anchor XF="--draft-lora $D0/h4tr/peft/a1.0000 --draft-tag --tag-set mono"; }
