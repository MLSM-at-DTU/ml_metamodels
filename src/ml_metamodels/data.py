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
        self.num_edge_features = 3
        self.num_node_features = 24
        # See if scaling is required
        self.scaling = self.cfg.scaling

        # Create paths based on the dataset name
        self.raw_data_path = os.path.join(self.raw_dir, cfg.data.dataset_name)
        # Ensure the raw directory exists
        assert osp.exists(self.raw_data_path), f"Raw data path {self.raw_data_path} does not exist."

        required_splits = ["train", "val", "test"]
        for split in required_splits:
            file_path = osp.join(self.raw_data_path, f"{split}.pickle")
            assert osp.exists(file_path), (
                f"Missing file: {split}.pickle in the specified path: {self.raw_data_path}. The expected format for the SiouxFalls24Zones dataset "
                f"is three separate datasets: 'train.pickle', 'val.pickle', and 'test.pickle', with each containing lists of PyG Data objects."
            )

    def add_identity_edge_features(self, graph):
        # Add one-hot encoding for edges
        num_edges = graph.edge_index.shape[1]
        one_hot_edges = torch.eye(num_edges, dtype=torch.float32)
        if graph.edge_attr is not None:
            # Concatenate one-hot encoding to existing edge_attr
            graph.edge_attr = torch.cat([graph.edge_attr, one_hot_edges], dim=1)
        else:
            # If edge_attr is None, initialize it with the one-hot encoding
            graph.edge_attr = one_hot_edges

        return graph

    def move_edge_feature_to_weight(self, graph):
        # Ensure the index is valid
        assert 0 <= self.edge_feature_to_weight < graph.edge_attr.shape[1], "Invalid edge feature index"

        # Extract the selected feature and assign it as edge_weight
        graph.edge_weight = graph.edge_attr[:, self.edge_feature_to_weight]

        # Remove the selected column from edge_attr
        if graph.edge_attr.shape[1] > 1:  # Only remove if more than one feature exists

            graph.edge_attr = torch.cat(
                [
                    graph.edge_attr[:, : self.edge_feature_to_weight],
                    graph.edge_attr[:, (self.edge_feature_to_weight + 1) :],
                ],
                dim=1,
            )
        else:
            graph.edge_attr = None  # If only one feature, set edge_attr to None

        # Ensure edge_weight is not all zeros
        assert torch.any(graph.edge_weight != 0), "Edge weight cannot be all zeros"

        return graph

    def save_data_and_scalers(self):
        torch.save(self.train_graphs, osp.join(self.processed_data_path, "train.pt"))
        torch.save(self.val_graphs, osp.join(self.processed_data_path, "val.pt"))
        torch.save(self.test_graphs, osp.join(self.processed_data_path, "test.pt"))

        if self.scaling is not None:
            with open(osp.join(self.processed_data_path, "scalers.pkl"), "wb") as f:
                pickle.dump({"node_scaler": self.node_scaler, "edge_scaler": self.edge_scaler}, f)

    def preprocess(self) -> None:
        """Preprocess the raw data and save it to the output folder."""

        # Create the processed data directory
        self.processed_data_path = os.path.join(self.processed_dir, self.cfg.data.dataset_name)
        # Ensure the processed directory exists
        os.makedirs(self.processed_data_path, exist_ok=True)
        self.edge_feature_to_weight = self.cfg.data.edge_feature_to_weight_index
        self.edge_feature_identity_matrix = self.cfg.data.edge_feature_identity_matrix

        # Preprocess the data
        for split in ["train", "val", "test"]:
            with open(osp.join(self.raw_data_path, f"{split}.pickle"), "rb") as f:
                graphs = pickle.load(f)

            preprocessed_graphs = []
            for graph in graphs:
                graph.x = graph.x.clone().detach().float()
                graph.edge_attr = graph.edge_attr.view(-1, self.num_edge_features).clone().detach().float()
                # Make sure 32 float
                graph.y = graph.y.clone().detach().float()
                graph.edge_weight = graph.edge_weight.clone().detach().float()

                if self.edge_feature_to_weight is not None:
                    graph = self.move_edge_feature_to_weight(graph)

                if self.edge_feature_identity_matrix:
                    if self.scaling is None:
                        graph = self.add_identity_edge_features(graph)
                    else:
                        Warning("Edge feature identity matrix will be added after scaling.")
                # Append the preprocessed graph
                preprocessed_graphs.append(graph)

            if split == "train":
                self.train_graphs = preprocessed_graphs
            elif split == "val":
                self.val_graphs = preprocessed_graphs
            elif split == "test":
                self.test_graphs = preprocessed_graphs

        # Preprocess the data
        if self.scaling is not None:
            print("Preprocessing data with scaling...")
            if self.edge_feature_to_weight:
                self.num_edge_features -= 1
            self._preprocess_with_scaling()

    def _preprocess_with_scaling(self) -> None:
        scaling_type = self.cfg.scaling

        if scaling_type == "StandardScaler":
            # Initialize scalers for node and edge features
            self.node_scaler = StandardScaler()
            self.edge_scaler = StandardScaler()
        else:
            raise NotImplementedError(f"Scaling type {scaling_type} is not implemented.")

        # Collect all node and edge features for training data
        all_node_features = []
        all_edge_features = []

        # Preprocess the data
        for split in ["train", "val", "test"]:
            graphs = getattr(self, f"{split}_graphs")

            if split == "train":
                for graph in graphs:
                    all_node_features.extend(graph.x.flatten().tolist())
                    all_edge_features.extend(graph.edge_attr.view(-1, self.num_edge_features).tolist())

                all_node_features_for_scalar = torch.tensor(all_node_features).clone().detach().reshape(-1, 1)
                all_edge_features_for_scalar = torch.tensor(all_edge_features).clone().detach()

                # Fit scalers using all training data
                self.node_scaler.fit(all_node_features_for_scalar)
                self.edge_scaler.fit(all_edge_features_for_scalar)

            normalized_graphs = []
            for graph in graphs:
                # Normalize using fitted scalers
                graph.x = torch.tensor(

                    self.node_scaler.transform(graph.x.flatten().reshape(-1, 1)).reshape(graph.x.shape),
                    dtype=torch.float32,
                )

                graph.edge_attr = torch.tensor(
                    self.edge_scaler.transform(graph.edge_attr.view(-1, self.num_edge_features)), dtype=torch.float32
                )

                # Add identity matrix to edge features
                if self.edge_feature_identity_matrix:
                    graph = self.add_identity_edge_features(graph)

                normalized_graphs.append(graph)

            if split == "train":
                self.train_graphs = normalized_graphs
            elif split == "val":
                self.val_graphs = normalized_graphs
            elif split == "test":
                self.test_graphs = normalized_graphs

@hydra.main(config_path="../../configs", config_name="gnn_config", version_base=None)
def main(cfg: DictConfig) -> None:
    data_class = cfg.data.data_class

    if data_class == "SiouxFalls24Zones":
        print("Preprocessing data...")
        dataset_object = SiouxFalls24Zones(cfg)
        dataset_object.preprocess()
        dataset_object.save_data_and_scalers()
        print("Preprocessing complete!")
    else:
        raise NotImplementedError(f"Data class {data_class}) is not implemented.")


if __name__ == "__main__":
    main()
