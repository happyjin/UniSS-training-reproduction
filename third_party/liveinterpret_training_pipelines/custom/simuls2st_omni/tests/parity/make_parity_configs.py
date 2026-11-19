"""生成对拍用的阶段配置：数据 / 采样取夹具（make_smoke_fixture.py）的配置，训练段取源项目配置，
只改步数、日志与评估间隔、dtype（CPU 上 fp32），并把路径写成绝对路径。

    python make_parity_configs.py --fixture /tmp/simuls2st_smoke --out /tmp/simuls2st_parity --sides src nat sw

每个 side 一套 `cfg/<side>/stage{1,2,3}.yaml`，输出到 `<out>/<side>/stageN`，阶段交接指向同一 side 的上一阶段。
swift 的上一阶段产物在 `<out>/sw/stageN/checkpoint-<最后一步>`，由 run_parity.sh 用环境变量覆盖。
"""
import argparse
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_CONFIGS = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'pipelines', 'simuls2st_omni', 'configs'))
# 夹具很小：Stage 2 只跑若干步（源配置是 23438 步）；评估间隔调小以覆盖 eval 路径
TRAIN_OVERRIDES = {
    'stage1': {'eval_steps': 5},
    'stage2': {'max_steps': 12},
    'stage3': {'eval_steps': 4},
}
COMMON = {'logging_steps': 1, 'save_steps': 1000, 'dataloader_num_workers': 0, 'bf16': False}


def _abs(data_cfg, root):
    for k, v in list(data_cfg.items()):
        if isinstance(v, dict):
            if k == 'roots':
                data_cfg[k] = {kk: os.path.join(root, vv) for kk, vv in v.items()}
            else:
                _abs(v, root)
        elif k == 'manifests' and isinstance(v, list):
            data_cfg[k] = [os.path.join(root, x) for x in v]
        elif k in ('root', 'si_root'):
            data_cfg[k] = os.path.join(root, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixture', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sides', nargs='+', default=['src', 'nat', 'sw'])
    ap.add_argument('--source_configs', default=SOURCE_CONFIGS)
    a = ap.parse_args()
    data_root = os.path.join(a.fixture, 'data')
    for side in a.sides:
        os.makedirs(os.path.join(a.out, 'cfg', side), exist_ok=True)
        for n, st in enumerate(('stage1', 'stage2', 'stage3'), start=1):
            fx = yaml.safe_load(open(os.path.join(a.fixture, 'configs', f'{st}.yaml')))
            src = yaml.safe_load(open(os.path.join(a.source_configs, f'{st}.yaml')))
            _abs(fx['data'], data_root)
            model = dict(src['model'])
            model['path'] = os.path.join(a.fixture, 'model')
            model['dtype'] = 'fp32'
            prev = os.path.join(a.out, side, f'stage{n - 1}')
            if st == 'stage2':
                model['init_talker_from'] = prev
            if st == 'stage3':
                model['merge_adapters_from'] = prev
            training = dict(src['training'])
            training.update(COMMON)
            training.update(TRAIN_OVERRIDES[st])
            if 'max_steps' in TRAIN_OVERRIDES[st]:
                training.pop('num_train_epochs', None)
            training['output_dir'] = os.path.join(a.out, side, st)
            cfg = {'stage': st, 'model': model, 'data': fx['data'], 'sampler': fx['sampler'], 'training': training}
            with open(os.path.join(a.out, 'cfg', side, f'{st}.yaml'), 'w') as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f'[parity] configs -> {a.out}/cfg/{{{",".join(a.sides)}}}/stage{{1,2,3}}.yaml')


if __name__ == '__main__':
    main()
