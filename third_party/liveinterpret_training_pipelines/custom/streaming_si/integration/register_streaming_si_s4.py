"""S4 的 swift 接线：注册 `streaming_si_s4` / `streaming_si_s4_val` 数据集（指针行，见 s4_pipeline）。"""
from datasets import Dataset as HfDataset
from swift.dataset import DatasetMeta, register_dataset
from swift.dataset.loader import DatasetLoader

from ..data.s4_pipeline import build_train_rows, build_val_rows, read_val_meta, s4_env

TRAIN_NAME = 'streaming_si_s4'
VAL_NAME = 'streaming_si_s4_val'


class S4PointerLoader(DatasetLoader):
    """把 `dataset_path` 解释为 S4 数据说明，产出指针行（见模块 docstring）。"""

    def _load_dataset_path(self, dataset_path, dataset_meta):
        cfg = s4_env()
        rows = build_val_rows(cfg) if dataset_meta.dataset_name == VAL_NAME else build_train_rows(cfg)
        if not rows['s4_index']:
            raise RuntimeError(f'{dataset_meta.dataset_name} 为空（检查 SI_S4_DATA / SI_S4_VAL_DIR）')
        return HfDataset.from_dict(rows)


def register_streaming_si_s4_datasets() -> None:
    cfg = s4_env()
    if not cfg['data']:
        return
    register_dataset(
        DatasetMeta(dataset_name=TRAIN_NAME, ms_dataset_id=TRAIN_NAME, dataset_path=cfg['data'],
                    loader=S4PointerLoader, tags=['audio', 'streaming-si', 's4']),
        exist_ok=True)
    if read_val_meta(cfg['val_dir']):
        register_dataset(
            DatasetMeta(dataset_name=VAL_NAME, ms_dataset_id=VAL_NAME, dataset_path=cfg['val_dir'],
                        loader=S4PointerLoader, tags=['audio', 'streaming-si', 's4']),
            exist_ok=True)
