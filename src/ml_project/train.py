import matplotlib.pyplot as plt
import torch
import typer
from torch_geometric.loader import DataLoader
from src.ml_project.model import GCN
import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import os.path as osp
import os
import numpy as np
import random
import wandb
from dotenv import load_dotenv


class TrainModel:
    def __init__(self, cfg: DictConfig) -> None:
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
        print(f"WANDB_API_KEY loaded successfully.")

        # Initialize wandb and set log directory
        self.wandb_run = wandb.init(
            project=self.cfg.wandb.project_name, config=OmegaConf.to_container(self.cfg, resolve=True)
        )

        # Save the processed data to wandb
        self._save_data_to_wandb()

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
            ).to(self.cfg.train.device)

        else:
            raise ValueError(f"Model type {self.model.layer_type} not supported. Please choose from ['GCN']")

        return model

    def train(self) -> None:
        """Train the GCN model on the Sioux Falls dataset."""
        # Check if the processed data path exists
        self._check_data_path()

        # Initialize wandb
        self._config_wandb()

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
        self.edge_feature_dim = train_data[0].edge_attr.shape[1]

        # Initialize model
        model = self.load_model()

        # Loss function and optimizer
        loss_fn = torch.nn.L1Loss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay
        )

        # Training loop
        statistics = {"train_loss": [], "val_loss": []}
        for epoch in range(self.cfg.train.epochs):
            model.train()
            epoch_loss = 0
            for data in train_loader:
                data = data.to(self.cfg.train.device)  # Moves to device first
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
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(self.cfg.train.device)
                    y_pred = model(data)
                    val_loss += loss_fn(y_pred, data.y).item()
                val_loss /= len(val_loader)
                statistics["val_loss"].append(val_loss)
                print(f"Epoch {epoch+1}/{self.cfg.train.epochs}, Loss: {epoch_loss}, Validation Loss: {val_loss}")
                wandb.log({"Epoch": epoch, "Train_loss": epoch_loss, "Valildation_loss": val_loss})

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

        # Plot training and validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(statistics["train_loss"], label="Train Loss")
        plt.plot(statistics["val_loss"], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        graph_name = "training_validation_loss.png"
        graph_dir = osp.join("reports/figures/" + graph_name)
        plt.savefig(graph_dir)


@hydra.main(config_path="../../configs", config_name="gnn_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Train the model
    trainer = TrainModel(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
