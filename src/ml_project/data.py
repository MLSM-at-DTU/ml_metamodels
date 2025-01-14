import typer
from torch.utils.data import Dataset
import os.path as osp
import pickle
import torch
import os
from sklearn.preprocessing import StandardScaler
import hydra
from omegaconf import DictConfig

class SiouxFalls24Zones(Dataset):
    """My custom dataset."""

    def __init__(self, cfg: DictConfig) -> None:
        # Load the configuration
        self.cfg = cfg
        # Define the base directories
        self.raw_dir = "data/raw"
        self.processed_dir = "data/processed"

        # Create paths based on the dataset name
        self.raw_data_path = os.path.join(self.raw_dir, cfg.data.dataset_name)
        # Ensure the raw directory exists
        assert osp.exists(self.raw_data_path), f"Raw data path {self.raw_data_path} does not exist."

        required_splits = ['train', 'val', 'test']
        for split in required_splits:
            file_path = osp.join(self.raw_data_path, f'{split}.pickle')
            assert osp.exists(file_path), (
                f"Missing file: {split}.pickle in the specified path: {self.raw_data_path}. The expected format for the SiouxFalls24Zones dataset "
                f"is three separate datasets: 'train.pickle', 'val.pickle', and 'test.pickle', with each containing lists of PyG Data objects."
            )


    def preprocess(self) -> None:
        """Preprocess the raw data and save it to the output folder."""

        # Create the processed data directory
        self.processed_data_path = os.path.join(self.processed_dir, self.cfg.data.dataset_name)
        # Ensure the processed directory exists
        os.makedirs(self.processed_data_path, exist_ok=True)

        # See if scaling is required
        scaling = self.cfg.scaling

        # Preprocess the data
        if scaling is None:
            print("Preprocessing data without scaling...")
            self._preprocess_no_scaling()
        else:
            print("Preprocessing data with scaling...")
            self._preprocess_with_scaling()

    def _preprocess_no_scaling(self) -> None:
        # Preprocess the data
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_data_path, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            processed_graphs = []
            for graph in graphs:
                # Normalize using fitted scalers
                graph.x = torch.tensor(graph.x.flatten().reshape(-1, 1), dtype=torch.float32)
                graph.edge_attr = torch.tensor(graph.edge_attr.view(-1, 3), dtype=torch.float32)

                # Add one-hot encoding for edges
                num_edges = graph.edge_attr.shape[0]
                one_hot_edges = torch.eye(num_edges, dtype=torch.float32)
                graph.edge_attr = torch.cat([graph.edge_attr, one_hot_edges], dim=1)

                # Make sure 32 float
                graph.y = graph.y.float()
                graph.edge_weight = graph.edge_weight.float()

                processed_graphs.append(graph)
            # Save processed graphs
            torch.save(processed_graphs, osp.join(self.processed_data_path, f'{split}.pt'))

    def _preprocess_with_scaling(self) -> None:
        scaling_type = self.cfg.scaling

        if scaling_type == 'StandardScaler':
            # Initialize scalers for node and edge features
            node_scaler = StandardScaler()
            edge_scaler = StandardScaler()
        else:
            raise NotImplementedError(f"Scaling type {scaling_type} is not implemented.")

        # Collect all node and edge features for training data
        all_node_features = []
        all_edge_features = []

        # Preprocess the data
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_data_path, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)
            if split == 'train':
                for graph in graphs:
                    all_node_features.extend(graph.x.flatten().tolist())
                    all_edge_features.extend(graph.edge_attr.view(-1, 3).tolist())

                all_node_features_for_scalar = torch.tensor(all_node_features).clone().detach().reshape(-1, 1)
                all_edge_features_for_scalar = torch.tensor(all_edge_features).clone().detach()

                # Fit scalers using all training data
                node_scaler.fit(all_node_features_for_scalar)
                edge_scaler.fit(all_edge_features_for_scalar)

            processed_graphs = []
            for graph in graphs:
                # Normalize using fitted scalers
                graph.x = torch.tensor(node_scaler.transform(graph.x.flatten().reshape(-1, 1)).reshape(graph.x.shape), dtype=torch.float32)
                graph.edge_attr = torch.tensor(edge_scaler.transform(graph.edge_attr.view(-1, 3)), dtype=torch.float32)

                # Add one-hot encoding for edges
                num_edges = graph.edge_attr.shape[0]
                one_hot_edges = torch.eye(num_edges, dtype=torch.float32)
                graph.edge_attr = torch.cat([graph.edge_attr, one_hot_edges], dim=1)

                # Make sure 32 float
                graph.y = graph.y.float()
                graph.edge_weight = graph.edge_weight.float()

                processed_graphs.append(graph)
            # Save processed graphs
            torch.save(processed_graphs, osp.join(self.processed_data_path, f'{split}.pt'))

        # Save scalers for future use
        with open(osp.join(self.processed_data_path, 'scalers.pkl'), 'wb') as f:
            pickle.dump({'node_scaler': node_scaler, 'edge_scaler': edge_scaler}, f)

@hydra.main(config_path="../../configs", config_name="gnn_config", version_base=None)
def main(cfg: DictConfig) -> None:
    data_class = cfg.data.data_class

    if data_class == 'SiouxFalls24Zones':
        print("Preprocessing data...")
        dataset_object = SiouxFalls24Zones(cfg)
        dataset_object.preprocess()
        print("Preprocessing complete!")
    else:
        raise NotImplementedError(f"Data class {data_class}) is not implemented.")

if __name__ == "__main__":
    main()
