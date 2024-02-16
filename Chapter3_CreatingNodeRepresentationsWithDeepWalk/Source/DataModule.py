from LibrariesModule import *


class KarateClubDataModule(L.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        # Set up path, transform, augmentation

    def prepare_data(self) -> None:
        # download, tokenize, save it to disk, etc...
        # don't use assign to self...

        return None
    
    def random_walk(self, G, node, walk_length):
        walk = [node]
        for _ in range(walk_length):
            neighbors = list(G.neighbors(node))
            if len(neighbors) == 0:
                break
            node = np.random.choice(neighbors)
            walk.append(node)
        return walk
    
    def setup(self, stage: str) -> None:
        # There are also data operations you might want to perform on every GPU
        # such as:
        # 1. count number of classes
        # 2. build vocabulary 
        # 3. perform train/val/test splits
        # 4. create datasets
        # 5. apply transforms (defined explicitly in your datamodule)
        self.train_dataset = KarateClub()


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=1, shuffle=True)
    
