import pytest
import os
from omegaconf import OmegaConf
from src.ml_project.train import TrainModel


@pytest.fixture
def test_cfg(tmp_path):
    """Create a test-specific configuration."""
    return OmegaConf.create(
        {
            "data": {
                "dataset_name": "sioux_falls_simulation_24_zones_OD_2K",
            },
            "train": {
                "random_seed": 42,
                "batch_size": 4,
                "lr": 0.001,
                "weight_decay": 1e-4,
                "device": "cpu",
                "epochs": 1,
            },
            "model": {
                "layer_type": "GCN",
                "hidden_dim": 64,
                "num_gnn_layers": 2,
            },
            "wandb": {
                "project_name": "TestProject",
            },
        }
    )


@pytest.mark.skipif(not os.path.exists("data/raw/sioux_falls_simulation_24_zones_OD_2K"), reason="Data files not found")
def test_train_init(test_cfg):  ### Not sure I like this one..
    """Test the _set_seed method."""
    trainer = TrainModel(test_cfg)

    assert isinstance(trainer, TrainModel)
