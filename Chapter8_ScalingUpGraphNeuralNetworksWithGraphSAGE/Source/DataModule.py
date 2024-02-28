from LibrariesModule import *


class PubMed(L.LightningDataModule):
    def __init__(self, data_root: str = 'DataFolder', 
                 batch_size: int = 4,
                 num_workers: int = 4,
                 num_neighbors: List[int] = [10, 10],
                 ) -> None:
        super().__init__()
        self.data_root      = data_root
        self.batch_size     = batch_size
        self.num_neighbors  = num_neighbors
        self.num_workers    = num_workers

    def prepare_data(self) -> None:
        # download and setting data
        # prepare_data is called from the main process. 
        # It is not recommended to assign state here 
        # (e.g. self.x = y) since it is called on a single process and 
        # if you assign states here then they won’t be available for other processes.
        
        Planetoid(root=self.data_root, name='Pubmed') # Download and save the dataset

    def setup(self, stage: str = '') -> None:
        # - count number of classes
        # - build vocabulary
        # - perform train/val/test splits
        # - create datasets
        # - apply transforms (defined explicitly in your datamodule)
        # - etc ...
        self.dataset = Planetoid(root=self.data_root, name='Pubmed')
        print(f'{20*"-"}')
        print(f'Here are data information:')
        print(f'Dataset: {self.dataset.name}')
        print(f'Number of graphs: {len(self.dataset)}')
        print(f'Number of nodes: {self.dataset._data.num_nodes}')
        print(f'Number of features: {self.dataset.num_features}')
        print(f'Number of classes: {self.dataset.num_classes}')

        print(f'Graph')
        print(f'{20*"-"}')
        print(f'Training nodes: {self.dataset[0].train_mask.sum().item()}')
        print(f'Validation nodes: {self.dataset[0].val_mask.sum().item()}')
        print(f'Test nodes: {self.dataset[0].test_mask.sum().item()}')
        print(f'Edge are directed: {self.dataset[0].is_directed()}')
        print(f'Graph has isolated nodes: {self.dataset[0].has_isolated_nodes()}')
        print(f'Graph has loops: {self.dataset[0].has_self_loops()}')


    def train_dataloader(self):
        return NeighborLoader(data=self.dataset[0], 
                              num_neighbors=self.num_neighbors,
                              batch_size=self.batch_size,
                              input_nodes=self.dataset[0].train_mask,
                              num_workers=self.num_workers)
    
    def val_dataloader(self):
        return NeighborLoader(data=self.dataset[0], 
                              num_neighbors=self.num_neighbors,
                              batch_size=self.batch_size,
                              input_nodes=self.dataset[0].val_mask,
                              num_workers=self.num_workers)
    
    def test_dataloader(self):
        return NeighborLoader(data=self.dataset[0], 
                              num_neighbors=self.num_neighbors,
                              batch_size=self.batch_size,
                              input_nodes=self.dataset[0].test_mask,
                              num_workers=self.num_workers)