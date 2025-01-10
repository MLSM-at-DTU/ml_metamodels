import matplotlib.pyplot as plt
import torch
import typer
from torch_geometric.loader import DataLoader
from ml_project.data import SiouxFalls24Zones
from ml_project.model import GCN, MLP
import datetime
import hydra
from omegaconf import DictConfig, OmegaConf 
import os.path as osp
import os
import numpy as np
import random
import wandb

class TrainModel():
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.run_dir = cfg.run_dir
        self._set_seed(cfg.train.random_seed)  # Set seed for reproducibility
        self.processed_data_path = osp.join(cfg.data.processed_dir, cfg.data.dataset_name)
        self._check_data_path()
        self._make_run_dir_folders()
        self._make_logger()
        self._save_data()
      
    def _make_run_dir_folders(self) -> None:
        # Check if run directory exists
        if not osp.exists(self.run_dir):
            raise FileNotFoundError(f"Run directory {self.run_dir} does not exist. Please create it first.")

        # Store model
        self.run_dir_model = osp.join(self.run_dir, "models")
        os.makedirs(self.run_dir_model, exist_ok=False)
        
        # Store figures
        self.run_dir_figures = osp.join(self.run_dir, "figures")
        os.makedirs(self.run_dir_figures, exist_ok=False)
        
        # Store wandb logs
        self.run_dir_wandb = osp.join(self.run_dir, "wandb")
        os.makedirs(self.run_dir_wandb, exist_ok=False)

        # Store dataset
        self.run_dir_data = osp.join(self.run_dir, "data")
        os.makedirs(self.run_dir_data, exist_ok=False)

    def _save_data(self) -> None:
        """Save the processed data."""
        
        for split in ["train", "val", "test"]:
            # Load data
            data_path = osp.join(self.processed_data_path, f'{split}.pt')
            data = torch.load(data_path, weights_only = False)

            # Save data
            data_dir = osp.join(self.run_dir_data, f'{split}.pt')
            torch.save(data, data_dir)
            artifact = wandb.Artifact(
                name=f'{split}.pt',
                type='data',
                description=f"Processed {split} data for the Sioux Falls network.")
            artifact.add_file(data_dir)
            self.wandb_run.log_artifact(artifact)


        return data
    def _make_logger(self) -> None:
        """Create a logger object."""
        # Initialize wandb and set log directory
        self.wandb_run = wandb.init(
            project=self.cfg.wandb.project_name,
            config=OmegaConf.to_container(self.cfg, resolve=True),
            name=self.run_dir,
            dir=self.run_dir_wandb)
        
    def _check_data_path(self) -> None:
        """Check if the processed data path exists."""
        self.training_data_path = osp.join(self.processed_data_path, 'train.pt')
        if not osp.exists(self.training_data_path):
            raise FileNotFoundError(f"Processed data path {self.training_data_path} does not exist. Please run the data processing script first.")
        
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

    def load_data(self, split = 'train') -> None:
        """Load a part of the data."""
        assert split in ['train', 'val'], "Only the splits 'train' and 'val' are supported."
        
        # Load data
        data_dir = osp.join(self.run_dir_data, f'{split}.pt')
        data = torch.load(data_dir, weights_only = False)
        return data
    
    def load_model(self) -> None:
        """Load a model architecture."""
        
        # # Load model configuration
        print("Model configuration:")
        print(OmegaConf.to_yaml(self.cfg.model))
        hidden_dim = self.cfg.model.hidden_dim
        device = self.device
        model_type = self.cfg.model.type
        num_gnn_layers = self.cfg.model.num_gnn_layers
        # Logging
        wandb.log({"model_type": model_type, "hidden_dim": hidden_dim, "node_feature_dim": self.node_feature_dim, "edge_feature_dim": self.edge_feature_dim})

        if model_type == 'GCN':
            model = GCN(node_feature_dim=self.node_feature_dim, 
                        edge_feature_dim = self.edge_feature_dim, 
                        hidden_dim=hidden_dim, 
                        num_gnn_layers = num_gnn_layers).to(device)
        
        elif model_type == 'MLP':
            model = MLP(node_feature_dim=self.node_feature_dim, 
                        edge_feature_dim = self.edge_feature_dim, 
                        hidden_dim=hidden_dim).to(device)
        else:
            raise ValueError(f"Model type {model_type} not supported. Please choose from ['GCN']")

        return model

    def train(self) -> None:
        """Train the GCN model on the Sioux Falls dataset."""
        # Load device configuration
        self.device = self.cfg.train.device
        
        # Load training configuration
        print("Training configuration:")
        print(OmegaConf.to_yaml(self.cfg.train))
        epochs = self.cfg.train.epochs
        batch_size = self.cfg.train.batch_size
        weight_decay = self.cfg.train.weight_decay
        lr = self.cfg.train.lr

        # Load training dataset
        train_data = self.load_data(split = 'train')
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, worker_init_fn=self._worker_init_fn)

        # Load validation dataset
        val_data = self.load_data(split = 'val')
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, worker_init_fn=self._worker_init_fn)

        # Get input dimension
        self.node_feature_dim = train_data[0].x.shape[1]
        self.edge_feature_dim = train_data[0].edge_attr.shape[1]
        
        # Initialize model
        model = self.load_model()

        # Loss function and optimizer
        loss_fn = torch.nn.L1Loss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Training loop
        statistics = {"train_loss": [], "val_loss": []}
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for data in train_loader:
                data = data.to(self.device)  # Moves to device first
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
                    data = data.to(self.device)
                    y_pred = model(data)
                    val_loss += loss_fn(y_pred, data.y).item()
                val_loss /= len(val_loader)
                statistics["val_loss"].append(val_loss)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}, Validation Loss: {val_loss}")
                wandb.log({"Epoch": epoch, "Train_loss": epoch_loss, "Valildation_loss": val_loss})

        # Save model
        print("Training complete")
        model_name = "model.pth"
        model_dir = osp.join(self.run_dir_model, model_name)
        torch.save(model.state_dict(), model_dir)

        # Artifact name
        artifact_name = str(self.run_dir).split("/")[-1] + f"_{model_name}"
        artifact = wandb.Artifact(
        name=artifact_name,
        type=self.cfg.model.type,
        description="A model trained to predict edge flow in the SiouxFalls network.",
        metadata={"L1 validation loss": val_loss})
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
        loss_graph = "training_loss.png"
        loss_graph_dir = osp.join(self.run_dir_figures, loss_graph)
        plt.savefig(loss_graph_dir)


@hydra.main(config_path="../../configs", config_name="gnn_config", version_base=None)
def main(cfg: DictConfig) -> None:
    
    # Set run directory
    OmegaConf.set_struct(cfg, False)          
    cfg.run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    OmegaConf.set_struct(cfg, True)

    # Train the model
    trainer = TrainModel(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
