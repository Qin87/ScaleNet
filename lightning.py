import pytorch_lightning as pl
import torch

class MyGNNLightning(pl.LightningModule):
    def __init__(self, model, train_mask, val_mask, test_mask, lr=0.01, weight_decay=0.001):
        super().__init__()
        self.model = model
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x, edge_index):
        return self.model(x, edge_index)

    def training_step(self, batch, batch_idx):
        x, edge_index, y = batch  # batch contains full features and edge_index
        out = self(x, edge_index)
        loss = self.criterion(out[self.train_mask], y[self.train_mask])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_index, y = batch
        out = self(x, edge_index)
        val_loss = self.criterion(out[self.val_mask], y[self.val_mask])
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # Scheduler: ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=80)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # ReduceLROnPlateau requires a monitored metric
            },
        }
