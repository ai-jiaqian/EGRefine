"""配置加载工具"""
import yaml


def load_config(path: str) -> dict:
    """加载 YAML 配置文件，返回 dict。"""
    with open(path, "r") as f:
        return yaml.safe_load(f)
