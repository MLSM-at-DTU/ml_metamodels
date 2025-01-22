import torch
from torch_geometric.loader import DataLoader
from ml_metamodels.model import GCN, GAT, DiffusionTestModel
import datetime
from hydra import initialize, compose
from omegaconf import OmegaConf
import os.path as osp
import os
import numpy as np
import random
import wandb
import typer
from typing import Dict, Any
from dotenv import load_dotenv

app = typer.Typer()


class TrainModel:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        processed_dir = "data/processed"
        self.processed_data_path = osp.join(processed_dir, cfg.data.dataset_name)
        self._set_seed(cfg.train.random_seed)  # Set seed for reproducibility

    def _config_wandb(self) -> None:
        """Create a logger object."""

        # Load .env file from root directory
        load_dotenv(".env")
        # Example: Access WANDB_API_KEY from the environment
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if not wandb_api_key:
            raise ValueError("WANDB_API_KEY is not set. Please check your .env file.")
        print("WANDB_API_KEY loaded successfully.")

        self.wandb_run = wandb.init()

    def _load_sweep_parameters(self):
        """
        Dynamically replace list values in the Hydra config with those from wandb.config for sweep.
        """
        # Convert Hydra config to a dictionary
        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)

        # Recursively update the config using flat keys from wandb.config
        def update_cfg_with_wandb(node, prefix=""):
            if isinstance(node, dict):
                for key, value in node.items():
                    current_path = f"{prefix}.{key}" if prefix else key
                    node[key] = update_cfg_with_wandb(value, current_path)
            elif isinstance(node, list):
                # Replace list with wandb.config value if available
                if prefix in wandb.config:
                    return wandb.config[prefix]
            return node

        # Update the config dictionary
        updated_cfg_dict = update_cfg_with_wandb(cfg_dict)

        # Convert back to OmegaConf for further use
        self.cfg = OmegaConf.create(updated_cfg_dict)
        self.wandb_run.config.update(OmegaConf.to_container(self.cfg, resolve=True))

    def _save_data_to_wandb(self) -> None:
        """Save the processed data into wandb."""
        # Save data to wandb
        for split in ["train", "val", "test"]:
            # Load data
            data_path = osp.join(self.processed_data_path, f"{split}.pt")
            artifact = wandb.Artifact(name=f"{split}.pt", type="data", description=f"Processed {split} data.")
            artifact.add_file(data_path)
            self.wandb_run.log_artifact(artifact)

    def _check_data_path(self) -> None:
        """Check if the processed data path exists."""
        self.training_data_path = osp.join(self.processed_data_path, "train.pt")
        if not osp.exists(self.training_data_path):
            raise FileNotFoundError(
                f"Processed data path {self.training_data_path} does not exist. Please run the data processing script first."
            )

    def _set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)

    def _worker_init_fn(worker_id) -> None:
        seed = torch.initial_seed() % (2**32)
        np.random.seed(seed)
        random.seed(seed)

    def load_data(self, split) -> None:
        """Load a part of the data."""

        # Ensure split used for training can only be train and validation
        assert split in ["train", "val"], "Only the splits 'train' and 'val' are supported."

        # Load the data
        data_path = osp.join(self.processed_data_path, f"{split}.pt")
        data = torch.load(data_path, weights_only=False)
        return data

    def load_model(self) -> None:
        """Load a model architecture."""

        # # Load model configuration
        print("Model configuration:")
        print(OmegaConf.to_yaml(self.cfg.model))

        if self.cfg.model.layer_type == "GCN":
            model = GCN(
                node_feature_dim=self.node_feature_dim,
                edge_feature_dim=self.edge_feature_dim,
                hidden_dim=self.cfg.model.hidden_dim,
                num_gnn_layers=self.cfg.model.num_gnn_layers,
                normalize=self.cfg.model.normalize,
                bias=self.cfg.model.bias,
                add_self_loops=self.cfg.model.add_self_loops,
            ).to(self.cfg.train.device)

        elif self.cfg.model.layer_type == "GAT":
            model = GAT(
                node_feature_dim=self.node_feature_dim,
                edge_feature_dim=self.edge_feature_dim,
                hidden_dim=self.cfg.model.hidden_dim,
                num_gnn_layers=self.cfg.model.num_gnn_layers,
            ).to(self.cfg.train.device)

        elif self.cfg.model.layer_type == "DiffusionTestModel":
            model = DiffusionTestModel(num_nodes=self.num_nodes, num_edges=self.num_edges).to(self.cfg.train.device)
        else:
            raise ValueError(f"Model type {self.cfg.model.layer_type} not supported. Please choose from ['GCN']")

        # Log model to wandb
        print("Model architecture: " + str(model))

        return model

    def _get_loss_fn(self) -> torch.nn.Module:
        """Get the loss function for the model."""

        loss_fn = self.cfg.train.loss_fn

        if loss_fn == "L1Loss":
            loss_fn = torch.nn.L1Loss()

        elif loss_fn == "MSELoss":
            loss_fn = torch.nn.MSELoss()

        else:
            raise ValueError(f"Loss function {loss_fn} not supported. Please choose from ['L1Loss', 'MSELoss']")

        return loss_fn

    def _get_validation_metrics(self, y_pred, y_true):
        """Get the validation metrics for the model."""

        # Calculate L1 loss
        l1_loss = torch.nn.L1Loss()
        l1_loss_value = l1_loss(y_pred, y_true).item()

        # Calculate MSE
        mse_loss = torch.nn.MSELoss()
        mse_loss_value_for_rmse = mse_loss(y_pred, y_true)
        mse_loss_value = mse_loss_value_for_rmse.item()

        # Calculate RMSE
        rmse_loss_value = torch.sqrt(mse_loss_value_for_rmse).item()

        return mse_loss_value, l1_loss_value, rmse_loss_value

    def train(self) -> None:
        """Train the GCN model on the Sioux Falls dataset."""
        # Check if the processed data path exists
        self._check_data_path()

        # Initialize wandb
        self._config_wandb()

        # Load sweep parameters into cfg
        self._load_sweep_parameters()

        # Save the processed data to wandb
        if self.cfg.wandb.save_data:
            self._save_data_to_wandb()

        # Load training configuration
        print("Training configuration:")
        print(OmegaConf.to_yaml(self.cfg.train))

        # Load training dataset
        train_data = self.load_data(split="train")
        train_loader = DataLoader(
            train_data, batch_size=self.cfg.train.batch_size, shuffle=True, worker_init_fn=self._worker_init_fn
        )

        # Load validation dataset
        val_data = self.load_data(split="val")
        val_loader = DataLoader(
            val_data, batch_size=self.cfg.train.batch_size, shuffle=False, worker_init_fn=self._worker_init_fn
        )

        # Get input dimension
        self.node_feature_dim = train_data[0].x.shape[1]
        self.num_nodes = train_data[0].x.shape[0]
        self.edge_feature_dim = train_data[0].edge_attr.shape[1]
        self.num_edges = train_data[0].edge_attr.shape[0]

        # Initialize model
        model = self.load_model()

        # Loss function and optimizer
        loss_fn = self._get_loss_fn()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay
        )

        # Training loop
        statistics = {"train_loss": [], "val_loss": []}
        for epoch in range(self.cfg.train.epochs):
            model.train()
            epoch_loss = 0
            for data in train_loader:
                data = data.to(self.cfg.train.device)
                optimizer.zero_grad()

                # Forward pass
                y_pred = model(data)
                loss = loss_fn(y_pred, data.y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Track statistics
            epoch_loss /= len(train_loader)
            statistics["train_loss"].append(epoch_loss)

            # Validate model
            model.eval()
            val_loss = 0
            mse_loss = 0
            l1_loss = 0
            rmse_loss = 0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(self.cfg.train.device)
                    y_pred = model(data)
                    val_loss += loss_fn(y_pred, data.y).item()

                    # Calculate validation metrics
                    mse, l1, rmse = self._get_validation_metrics(y_pred, data.y)
                    mse_loss += mse
                    l1_loss += l1
                    rmse_loss += rmse

                val_loss /= len(val_loader)
                mse_loss /= len(val_loader)
                l1_loss /= len(val_loader)
                rmse_loss /= len(val_loader)

                statistics["val_loss"].append(val_loss)
                print(
                    f"Epoch {epoch + 1}/{self.cfg.train.epochs}, Loss: {epoch_loss}, Validation Loss: {val_loss}, MSE loss: {mse_loss}, L1 Loss: {l1_loss}, RMSE Loss: {rmse_loss}"
                )
                wandb.log(
                    {
                        "Epoch": epoch,
                        "Train_loss": epoch_loss,
                        "Validation_loss": val_loss,
                        "L1_loss": l1_loss,
                        "MSE_loss": mse_loss,
                        "RMSE_loss": rmse_loss,
                    }
                )

        # Save model
        print("Training complete")
        model_name = f"{self.cfg.model.layer_type}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"
        model_dir = osp.join("models/", model_name)
        torch.save(model.state_dict(), model_dir)

        # Artifact name
        artifact_name = model_name
        artifact = wandb.Artifact(
            name=artifact_name,
            type=self.cfg.model.layer_type,
            description="A model trained to predict edge flow in the SiouxFalls network.",
            metadata={"L1 validation loss": val_loss},
        )
        artifact.add_file(model_dir)
        self.wandb_run.log_artifact(artifact)


def generate_sweep_configuration(
    cfg: Dict[str, Any],
    sweep_name: str = "sweep",
    metric_name: str = "L1_loss",
    goal: str = "minimize",
    method: str = "random",
) -> Dict[str, Any]:
    sweep_parameters = {}

    # Recursively find list-type entries in the config
    def find_list_entries(node, prefix=""):
        if isinstance(node, dict):
            for key, value in node.items():
                current_path = f"{prefix}.{key}" if prefix else key
                find_list_entries(value, current_path)
        elif isinstance(node, list):
            # Add the list as a sweep parameter
            sweep_parameters[prefix] = {"values": node}

    # Start processing the configuration
    find_list_entries(OmegaConf.to_container(cfg, resolve=True))

    # Create the sweep configuration
    sweep_configuration = {
        "method": method,
        "name": sweep_name,
        "metric": {"goal": goal, "name": metric_name},
        "parameters": sweep_parameters,
    }

    return sweep_configuration


def main() -> None:
    with initialize(config_path="../../configs"):
        # hydra.main() decorator was not used since it was conflicting with typer decorator
        cfg = compose(config_name="hydra_config.yaml")
    # Train the model
    trainer = TrainModel(cfg)
    trainer.train()


@app.command()
def run_training() -> None:
    with initialize(config_path="../../configs"):
        # Load the Hydra configuration
        cfg = compose(config_name="hydra_config.yaml")

    if cfg.wandb.sweep.enabled:
        # Generate the sweep configuration dynamically
        sweep_cfg = generate_sweep_configuration(
            cfg,
            sweep_name=cfg.wandb.sweep.sweep_name,
            metric_name=cfg.wandb.sweep.metric_name,
            goal=cfg.wandb.sweep.metric_goal,
            method=cfg.wandb.sweep.method,
        )

        # Check if the sweep_cfg contains any lists
        if not sweep_cfg["parameters"]:
            raise ValueError(
                "No list-type parameters found in the configuration. Please add at least one list-type parameter to run a sweep."
            )

        # Initialize W&B sweep
        sweep_id = wandb.sweep(sweep=sweep_cfg, project=cfg.wandb.project_name)

        # Start the sweep
        wandb.agent(sweep_id, function=main)

    elif not cfg.wandb.sweep.enabled:
        main()


if __name__ == "__main__":
    run_training()
