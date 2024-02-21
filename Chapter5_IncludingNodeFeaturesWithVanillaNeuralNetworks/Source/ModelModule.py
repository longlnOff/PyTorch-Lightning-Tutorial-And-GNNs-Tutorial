from LibrariesModule import *


class SimpleMLP(L.LightningModule):
    def __init__(self, dim_in, dim_hidden, dim_out) -> None:
        super().__init__()
        self.mlp1 = Linear(dim_in, dim_hidden)
        self.mlp2 = Linear(dim_hidden, dim_out)

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.mlp1(x)
        x = torch.relu(x)
        x = self.mlp2(x)

        return x
    
    def _common_step(self, data, mode='train'):
        x = data.x
        y = data.y
        x = self.forward(x)

        if mode == 'train':
            mask = data.train_mask
        elif mode == 'val':
            mask = data.val_mask
        elif mode == 'test':
            mask = data.test_mask
        else:
            assert False, "Unknown mode"
        
        loss = self.loss_fn(x[mask], y[mask])

        accuracy = (x[mask].argmax(dim=-1) == y[mask]).sum().float() / mask.sum()

        return loss, accuracy
    
    def training_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch, mode='train')
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', accuracy, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch, mode='val')
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch, mode='test')
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_accuracy', accuracy, prog_bar=True)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=2e-3)
    