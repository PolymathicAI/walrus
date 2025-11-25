import os.path as osp
import pathlib

import pytest
import torch
from hydra import compose, initialize

from walrus.train import CONFIG_DIR, CONFIG_NAME, main

from .utils import generate_parameters

# Set the different options to test
conf_options = {
    "trainer.prediction_type": ["delta", "full"],
    "trainer.enable_amp": ["False"],
    "model.causal_in_time": ["True", "False"],
}


@pytest.mark.parametrize("conf", generate_parameters(conf_options), indirect=True)
def test_train(conf):
    """Test training terminates normally for different sets of config."""
    overrides = conf
    cfg_dir = osp.relpath(CONFIG_DIR, pathlib.Path(__file__).resolve().parent)

    with initialize(config_path=str(cfg_dir)):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        with torch.autograd.detect_anomaly():
            main(cfg)
        assert True
