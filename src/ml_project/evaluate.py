import torch
import typer
from torch_geometric.loader import DataLoader
from data import SiouxFalls_24_zones
from src.ml_project.model import GCN

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DEVICE = torch.device("cpu")


def evaluate(model_checkpoint: str, batch_size: int = 32) -> None:
    """Evaluate a trained GCN model."""
    print("Evaluating GCN model")
    print(f"Model checkpoint: {model_checkpoint}")

    # Load dataset
    dataset = SiouxFalls_24_zones("sioux_falls_simulation_24_zones_OD_2K")
    test_data = dataset.load_split("test")
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Load model
    model = GCN(input_dim=24, hidden_dim=64).to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE, weights_only=False))

    model.eval()
    total_loss = 0
    loss_fn = torch.nn.L1Loss()

    # Evaluate
    with torch.no_grad():
        for data in test_loader:
            data = data.to(DEVICE)
            y_pred = model(data)
            loss = loss_fn(y_pred, data.y)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss}")


if __name__ == "__main__":
    typer.run(evaluate)
