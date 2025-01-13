import os.path as osp
import pickle
import torch
import pytest
import os

@pytest.mark.skipif(not os.path.exists("data/raw/sioux_falls_simulation_24_zones_OD_2K"), reason="Data files not found")
class TestSiouxFalls24Zones:

    def test_raw_data(self):
        """Test whether the dataset nodes and edges are correct."""
        
        raw_data_path = "data/raw/sioux_falls_simulation_24_zones_OD_2K"

        all_graphs = []
        # Preprocess the data
        for split in ['train', 'val', 'test']:
            with open(osp.join(raw_data_path, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)
                all_graphs.extend(graphs)
            assert graphs[0].x.shape == (24, 24)
            assert graphs[0].edge_attr.shape == (228,) ### raw data should be changed to 76 x 3 shape
            assert graphs[0].y.shape == (76,)
            assert graphs[0].edge_weight.shape == (76,)
        
        assert len(all_graphs) == 2000 ### Not sure I like to do this test if the number of graphs is not fixed

    def test_preprocessed_data(self):
        '''Test whether the preprocessed data is correct.'''

        preprocessed_data_path = "data/processed/sioux_falls_simulation_24_zones_OD_2K"

        # Preprocess the data
        for split in ['train', 'val', 'test']:
            graphs = torch.load(osp.join(preprocessed_data_path, f'{split}.pt'), weights_only=False)

        assert graphs[0].x.shape == (24, 24)
        assert graphs[0].edge_attr.shape == (76, 3)
        assert graphs[0].y.shape == (76,)
        assert graphs[0].edge_weight.shape == (76,)


    