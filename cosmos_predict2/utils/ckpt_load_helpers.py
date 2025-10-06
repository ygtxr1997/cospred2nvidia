from pathlib import Path
from imaginaire.utils import log


def load_config_from_checkpoint_dir(dit_path: str):
    """从检查点路径中提取配置文件夹并加载配置文件，优先使用 pkl"""
    dit_path_obj = Path(dit_path)
    checkpoint_dir = dit_path_obj.parent.parent.parent

    # 优先查找 pkl 文件
    config_pkl_path = checkpoint_dir / "config.pkl"

    if config_pkl_path.exists():
        import pickle
        log.info(f"Found pkl config file: {config_pkl_path}")
        with open(config_pkl_path, 'rb') as f:
            config_dict = pickle.load(f)
        return config_dict, config_pkl_path
    else:
        raise FileNotFoundError(f"Neither config.pkl nor config.yaml found in {checkpoint_dir}")

