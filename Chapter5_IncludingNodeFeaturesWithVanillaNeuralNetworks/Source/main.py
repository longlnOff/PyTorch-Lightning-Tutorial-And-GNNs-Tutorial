if __name__ == '__main__':
    from DataModule import *
    from LibrariesModule import *
    from ModelModule import *

    # To properly utilize Tensor Cores
    torch.set_float32_matmul_precision('high')

    trainer = L.Trainer(
        accelerator="auto",
        max_epochs=50,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )
    
    data_module = CoraData()
    data_module.setup()

    dim_in = data_module.dataset.num_features
    dim_out = data_module.dataset.num_classes
    dim_hidden = 16
    model = SimpleMLP(dim_in=dim_in, 
                      dim_hidden=dim_hidden, 
                      dim_out=dim_out)
    
    trainer.fit(model=model,
                datamodule=data_module)
    
    trainer.validate(model=model,
                     datamodule=data_module)
    
    trainer.test(model=model,
                 datamodule=data_module)