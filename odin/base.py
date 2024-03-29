from hydra import compose, initialize

with initialize(version_base=None, config_path="../config"):
    CONFIG = compose(config_name="main.yaml")
