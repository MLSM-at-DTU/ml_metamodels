import torch
import typer
from torch_geometric.loader import DataLoader
from omegaconf import OmegaConf
from hydra import initialize, compose
import wandb
from ml_metamodels.model import GCN, GAT
import os

app = typer.Typer()

class EvaluateModel:
    def __init__(self, cfg):
        self.wandb_run = cfg.inference.wandb_run

        self.load_wandb_config()
        self.load_data()
        self.load_model()

    def load_wandb_config(self):
        api = wandb.Api()
        run = api.run(f"{self.wandb_run}")
        self.cfg = OmegaConf.create(run.config)
        self.wandb_project = self.cfg.wandb.project_name

        # Identify the model artifact
        artifacts = list(run.logged_artifacts())
        artifact_types = [artifact.type for artifact in artifacts]
        if len(artifact_types) == 0:
            raise ValueError(f"No artifacts found in wandb run {self.wandb_run}")
        elif len(artifact_types) > 1:
            print("Multiple artifact types found in wandb run.")
            if "data" in artifact_types:
                print("Removing data types from artifact types.")
                artifact_types.remove("data")
                if len(artifact_types) > 1:
                    raise ValueError(f"Multiple artifact types found in wandb run {self.wandb_run} after removing data types.")                
        
        # Get the model artifact
        artifact_names = [artifact.name for artifact in artifacts]
        self.model_artifact = artifact_names[0]
        print(f"Model artifact: {self.model_artifact}")

        # Downaload the model checkpoint
        artifact = api.artifact(f"{self.wandb_project}/{self.model_artifact}")  # Use full artifact path
        self.model_checkpoint = artifact.download()
        self.model_checkpoint = os.path.join(self.model_checkpoint, self.model_artifact.split(":")[0])
        print(f"Model checkpoint downloaded to: {self.model_checkpoint}")

    def load_data(self):
        self.processed_dir = "data/processed"
        self.processed_data_path = os.path.join(self.processed_dir, self.cfg['data']['dataset_name'], "test.pt")

        # Load the test data
        self.dataset = torch.load(self.processed_data_path, weights_only=False)
        self.test_loader = DataLoader(self.dataset, batch_size=self.cfg['train']['batch_size'], shuffle=False)
        print(f"Loaded test data from {self.processed_data_path}")

        # Get the feature dimensions
        self.node_feature_dim = self.dataset[0].x.shape[1]
        self.edge_feature_dim = self.dataset[0].edge_attr.shape[1]
        

    def load_model(self):
        self.model_type = self.cfg.model.layer_type

        if self.model_type == "GCN":
            print("Loading GCN model")
            self.model = GCN(
                node_feature_dim=self.node_feature_dim,
                edge_feature_dim=self.edge_feature_dim,
                hidden_dim=self.cfg.model.hidden_dim,
                num_gnn_layers=self.cfg.model.num_gnn_layers,
                normalize = self.cfg.model.normalize, 
                bias = self.cfg.model.bias,
                add_self_loops = self.cfg.model.add_self_loops,
            ).to(self.cfg.train.device)

        elif self.model_type == "GAT":
            print("Loading GAT model")
            self.model = GAT(
                node_feature_dim=self.node_feature_dim,
                edge_feature_dim=self.edge_feature_dim,
                hidden_dim=self.cfg.model.hidden_dim,
                num_gnn_layers=self.cfg.model.num_gnn_layers,
            ).to(self.cfg.train.device)

        else:
            raise ValueError(f"Model type {self.model_type} not supported.")
        

    def loss_fn(self) -> torch.nn.Module:
        """Get the loss function for the model."""

        loss_fn = self.cfg.train.loss_fn

        if loss_fn == "L1Loss":
            loss_fn = torch.nn.L1Loss()

        elif loss_fn == "MSELoss":
            loss_fn = torch.nn.MSELoss()

        else:
            raise ValueError(f"Loss function {loss_fn} not supported. Please choose from ['L1Loss', 'MSELoss']")

        return loss_fn

    def evaluate(self) -> None:
        """Evaluate a trained GCN model."""
        
        print("Evaluating GCN model")
        print(f"Model checkpoint: {self.model_checkpoint}")

        # Load model
        self.model.load_state_dict(torch.load(self.model_checkpoint, map_location=self.cfg.train.device, weights_only=False))

        self.model.eval()
        total_loss = 0
        loss_fn = self.loss_fn()

        # Evaluate
        with torch.no_grad():
            for data in self.test_loader:
                data = data.to(self.cfg.train.device)
                y_pred = self.model(data)
                loss = loss_fn(y_pred, data.y)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.test_loader)
        print(f"Test Loss: {avg_loss}")

@app.command()
def main():
    with initialize(config_path="../../configs"):
        # Load the Hydra configuration
        cfg = compose(config_name="config.yaml")

    evaluate = EvaluateModel(cfg)
    evaluate.evaluate()

if __name__ == "__main__":
    app()
