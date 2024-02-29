from LibrariesModule import *


class PROTEINS(L.LightningDataModule):
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
        
        TUDataset(root=self.data_root, name='PROTEINS') # Download and save the dataset

    def setup(self, stage: str = '') -> None:
        # - count number of classes
        # - build vocabulary
        # - perform train/val/test splits
        # - create datasets
        # - apply transforms (defined explicitly in your datamodule)
        # - etc ...
        self.dataset = TUDataset(root=self.data_root, name='PROTEINS').shuffle()
        print(f'{20*"-"}')
        print(f'Here are data information:')
        print(f'Dataset: {self.dataset.name}')
        print(f'Number of graphs: {len(self.dataset)}')
        print(f'Number of nodes: {self.dataset._data.num_nodes}')
        print(f'Number of features: {self.dataset.num_features}')
        print(f'Number of classes: {self.dataset.num_classes}')

        print(f'Graph')
        print(f'{20*"-"}')
        print(f'Edge are directed: {self.dataset[0].is_directed()}')
        print(f'Graph has isolated nodes: {self.dataset[0].has_isolated_nodes()}')
        print(f'Graph has loops: {self.dataset[0].has_self_loops()}')

        self.train_dataset = self.dataset[:int(len(self.dataset) * 0.8)]
        self.val_dataset = self.dataset[int(len(self.dataset) * 0.8):int(len(self.dataset) * 0.9)]
        self.test_dataset = self.dataset[int(len(self.dataset) * 0.9):]

        print(f'training set: {len(self.train_dataset)} graphs')
        print(f'validation set: {len(self.val_dataset)} graphs')
        print(f'test set: {len(self.test_dataset)} graphs')
        

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.num_workers,
                            pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers,
                            pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers,
                            pin_memory=True)
    