# v11 P0-4 登记：规则选定的 stage-2 第 4000 步在测试集上解码一次（2026-11-15，测试解码之前写）

- 目的：回应模拟审稿 M1(a)。v10 报告的是用户按"BLEU 优先"改选的第 3500 步；登记规则选的是第 4000 步，从未在测试集上解码。
- 系统：joint v2′ stage 2 第 4000 步（ckpt/step_04000，DONE 存在），δ_w = −1（与 3500 相同，dev5 标定的值，不重新标定），
  base、Talker（zh→en 现有 PL-EM；En→Zh export_i0400）、读级合并、chunk 协议、两个 TTS 种子，全部与 wb_test.sh 相同。
- 测试集：AMI zh→en、AliMeeting、AMI En→Zh，各解码一次。脚本 experiments/mist_w3po_v1/scripts/wb_test_s4000.sh（wb_test.sh 只换检查点 / 名字 / 输出路径）。
- 指标：文本 tcpBLEU all/ov/1spk、chunk 文本 LAAL、ASR-BLEU、语音 LAAL（spread、block-start）、speech start、EndOffset、语速、停顿、UTMOS（post_review 工具）。
- 比较（全部描述性，不做任何选择）：vs 严格基线 B（gate_2seed）；vs 3500 + δ=−1（gate_2seed）。论文中作为附录行放在 3500 旁边；主系统（B+PA）不受影响。
- 不做：任何参数调整；任何基于该结果的检查点替换。
