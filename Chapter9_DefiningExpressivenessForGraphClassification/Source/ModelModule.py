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
        self.log(name='train_loss', 
                 value=loss, 
                 batch_size=batch.batch.shape[0],
                 prog_bar=True)
        
        self.log(name='train_accuracy',
                 value=accuracy,
                 batch_size=batch.batch.shape[0],
                 prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch, mode='val')
        self.log(name='val_loss', 
                 value=loss, 
                 batch_size=batch.batch.shape[0],
                 prog_bar=True)
        self.log(name='val_accuracy',
                 value=accuracy,
                 batch_size=batch.batch.shape[0],
                 prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch, mode='test')
        self.log(name='test_loss', 
                 value=loss, 
                 batch_size=batch.batch.shape[0],
                 prog_bar=True)
        self.log(name='test_accuracy',
                 value=accuracy,
                 batch_size=batch.batch.shape[0],
                 prog_bar=True)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=2e-3)
    


# ####################### GNN Model #######################
class VanillaGNNLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = Linear(dim_in, dim_out, bias=False)
    
    def forward(self, x, adjacency):
        x = self.linear(x)
        x = torch.sparse.mm(adjacency, x)
        return x


class SimpleGNN(L.LightningModule):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()

        self.gnn1 = VanillaGNNLayer(dim_in, dim_hidden)
        self.gnn2 = VanillaGNNLayer(dim_hidden, dim_out)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, adjacency):
        x = self.gnn1(x, adjacency)
        x = torch.relu(x)
        x = self.gnn2(x, adjacency)
        return torch.nn.functional.log_softmax(x, dim=1)
    
    def _common_step(self, data, mode='train'):
        x = data.x
        y = data.y
        adjacency = to_dense_adj(data.edge_index)[0]
        adjacency = adjacency + torch.eye(adjacency.shape[0], device=adjacency.device)
        x = self.forward(x, adjacency)

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
        self.log(name='train_loss', 
                 value=loss, 
                 batch_size=batch.batch.shape[0],
                 prog_bar=True)
        
        self.log(name='train_accuracy',
                 value=accuracy,
                 batch_size=batch.batch.shape[0],
                 prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch, mode='val')
        self.log(name='val_loss', 
                 value=loss, 
                 batch_size=batch.batch.shape[0],
                 prog_bar=True)
        self.log(name='val_accuracy',
                 value=accuracy,
                 batch_size=batch.batch.shape[0],
                 prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch, mode='test')
        self.log(name='test_loss', 
                 value=loss, 
                 batch_size=batch.batch.shape[0],
                 prog_bar=True)
        self.log(name='test_accuracy',
                 value=accuracy,
                 batch_size=batch.batch.shape[0],
                 prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=2e-3)
    


class SimpleGCN(L.LightningModule):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()

        self.gcn1 = GCNConv(dim_in, dim_hidden)
        self.gcn2 = GCNConv(dim_hidden, dim_out)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)
        return torch.nn.functional.log_softmax(x, dim=1)
    
    def _common_step(self, data, mode='train'):
        x = data.x
        y = data.y
        edge_index = data.edge_index
        x = self.forward(x=x, edge_index=edge_index)

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
        self.log(name='train_loss', 
                 value=loss, 
                 batch_size=batch.batch.shape[0],
                 prog_bar=True)
        
        self.log(name='train_accuracy',
                 value=accuracy,
                 batch_size=batch.batch.shape[0],
                 prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch, mode='val')
        self.log(name='val_loss', 
                 value=loss, 
                 batch_size=batch.batch.shape[0],
                 prog_bar=True)
        self.log(name='val_accuracy',
                 value=accuracy,
                 batch_size=batch.batch.shape[0],
                 prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch, mode='test')
        self.log(name='test_loss', 
                 value=loss, 
                 batch_size=batch.batch.shape[0],
                 prog_bar=True)
        self.log(name='test_accuracy',
                 value=accuracy,
                 batch_size=batch.batch.shape[0],
                 prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=2e-3)
    

class SimpleGATv2(L.LightningModule):
    def __init__(self, dim_in, dim_hidden, dim_out, num_heads):
        super().__init__()

        self.gat1 = GATv2Conv(in_channels=dim_in, out_channels=dim_hidden, heads=num_heads)
        self.gat2 = GATv2Conv(in_channels=dim_hidden * num_heads, out_channels=dim_out, heads=1)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x, edge_index):
        h = torch.nn.functional.dropout(input=x, p=0.6, training=self.training)
        h = self.gat1(h, edge_index)
        h = torch.nn.functional.elu(h)
        h = torch.nn.functional.dropout(input=h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index)
        return torch.nn.functional.log_softmax(h, dim=1)
    
    def _common_step(self, data, mode='train'):
        x = data.x
        y = data.y
        edge_index = data.edge_index
        x = self.forward(x=x, edge_index=edge_index)

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
        self.log(name='train_loss', 
                 value=loss, 
                 batch_size=batch.batch.shape[0],
                 prog_bar=True)
        
        self.log(name='train_accuracy',
                 value=accuracy,
                 batch_size=batch.batch.shape[0],
                 prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch, mode='val')
        self.log(name='val_loss', 
                 value=loss, 
                 batch_size=batch.batch.shape[0],
                 prog_bar=True)
        self.log(name='val_accuracy',
                 value=accuracy,
                 batch_size=batch.batch.shape[0],
                 prog_bar=True)
        

    def test_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch, mode='test')
        self.log(name='test_loss', 
                 value=loss, 
                 batch_size=batch.batch.shape[0],
                 prog_bar=True)
        self.log(name='test_accuracy',
                 value=accuracy,
                 batch_size=batch.batch.shape[0],
                 prog_bar=True)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=2e-3)
    

class SimpleGraphSAGE(L.LightningModule):
    def __init__(self, dim_in, dim_hidden, dim_out, aggr):
        super().__init__()

        self.sage1 = SAGEConv(in_channels=dim_in, out_channels=dim_hidden, aggr=aggr)
        self.sage2 = SAGEConv(in_channels=dim_hidden, out_channels=dim_out, aggr=aggr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index)
        h = torch.relu(h)
        h = torch.nn.functional.dropout(input=h, p=0.5, training=self.training)
        h = self.sage2(h, edge_index)
        return torch.nn.functional.log_softmax(h, dim=1)
    
    def _common_step(self, data, mode='train'):
        x = data.x
        y = data.y
        edge_index = data.edge_index
        x = self.forward(x=x, edge_index=edge_index)

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
        self.log(name='train_loss', 
                 value=loss, 
                 batch_size=batch.batch_size,
                 prog_bar=True)
        
        self.log(name='train_accuracy',
                 batch_size=batch.batch_size,
                 value=accuracy,
                 prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch, mode='val')
        self.log(name='val_loss', 
                 batch_size=batch.batch_size,
                 value=loss, 
                 prog_bar=True)
        self.log(name='val_accuracy',
                 batch_size=batch.batch_size,
                 value=accuracy,
                 prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch, mode='test')
        self.log(name='test_loss', 
                 batch_size=batch.batch_size,
                 value=loss, 
                 prog_bar=True)
        self.log(name='test_accuracy',
                 batch_size=batch.batch_size,
                 value=accuracy,
                 prog_bar=True)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=2e-3)
    


class SimpleGIN(L.LightningModule):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.conv1 = GINConv(
            torch.nn.Sequential(
                Linear(dim_in, dim_hidden),
                BatchNorm1d(dim_hidden),
                torch.nn.ReLU(),
                Linear(dim_hidden, dim_hidden),
                torch.nn.ReLU()
            )
        )

        self.conv2 = GINConv(
            torch.nn.Sequential(
                Linear(dim_hidden, dim_hidden),
                BatchNorm1d(dim_hidden),
                torch.nn.ReLU(),
                Linear(dim_hidden, dim_hidden),
                torch.nn.ReLU()
            )
        )

        self.conv3 = GINConv(
            torch.nn.Sequential(
                Linear(dim_hidden, dim_hidden),
                BatchNorm1d(dim_hidden),
                torch.nn.ReLU(),
                Linear(dim_hidden, dim_hidden),
                torch.nn.ReLU()
            )
        )

        self.linear1 = Linear(dim_hidden*3, dim_hidden*3)
        self.linear2 = Linear(dim_hidden*3, dim_out)

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x, edge_index, batch):
        # Node embeddings
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout (global pooling)
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat([h1, h2, h3], dim=1)

        # MLP
        h = self.linear1(h)
        h = torch.relu(h)
        # drop out
        h = torch.nn.functional.dropout(h, p=0.5, training=self.training)
        h = self.linear2(h)

        return torch.nn.functional.log_softmax(h, dim=1)
    
    def _common_step(self, data):
        x = data.x
        y = data.y
        edge_index = data.edge_index
        batch = data.batch
        out = self.forward(x=x, edge_index=edge_index, batch=batch)

        loss = self.loss_fn(out, y)
        accuracy = (out.argmax(dim=1) == y).sum().float() / y.shape[0]

        return loss, accuracy
    
    def training_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch)
        self.log(name='train_loss', 
                 value=loss, 
                 batch_size=batch.batch_size,
                 prog_bar=True)
        
        self.log(name='train_accuracy',
                 batch_size=batch.batch_size,
                 value=accuracy,
                 prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch)
        self.log(name='val_loss', 
                 batch_size=batch.batch_size,
                 value=loss, 
                 prog_bar=True)
        self.log(name='val_accuracy',
                 batch_size=batch.batch_size,
                 value=accuracy,
                 prog_bar=True)
        

    def test_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch)
        self.log(name='test_loss', 
                 batch_size=batch.batch_size,
                 value=loss, 
                 prog_bar=True)
        self.log(name='test_accuracy',
                 batch_size=batch.batch_size,
                 value=accuracy,
                 prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=2e-3)