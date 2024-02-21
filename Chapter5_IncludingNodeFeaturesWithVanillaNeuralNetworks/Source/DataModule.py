from LibrariesModule import *


class CoraData(L.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()

    def prepare_data(self) -> None:
        # download and setting data
        # prepare_data is called from the main process. 
        # It is not recommended to assign state here 
        # (e.g. self.x = y) since it is called on a single process and 
        # if you assign states here then they won’t be available for other processes.
        
        Planetoid(root='data', name='Cora') # Download and save the dataset
        pass

    def setup(self, stage: str = '') -> None:
        # - count number of classes
        # - build vocabulary
        # - perform train/val/test splits
        # - create datasets
        # - apply transforms (defined explicitly in your datamodule)
        # - etc ...
        self.dataset = Planetoid(root='data', name='Cora')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset, batch_size=64, num_workers=11)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset, batch_size=64, num_workers=11)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset, batch_size=1, num_workers=11)