if __name__ == '__main__':
    from DataModule import *
    from LibrariesModule import *
    from ModelModule import (
        SimpleMLP,
        SimpleGNN
    )
    from lightning.pytorch.cli import LightningCLI

    # To properly utilize Tensor Cores
    torch.set_float32_matmul_precision('high')

    cli = LightningCLI(datamodule_class=CoraData)
                       