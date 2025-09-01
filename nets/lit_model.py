import torch
from torch import nn, optim
import pytorch_lightning as pl


from torch.utils.data import Dataset
class FullBatchGraphDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        # There is only one sample which is the whole graph
        return 1

    def __getitem__(self, idx):
        return self.data



class Lit(pl.LightningModule):
    def __init__(self, model, args, train_mask, val_mask, test_mask, evaluator=None):
        super().__init__()
        self.model = model
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.evaluator = evaluator
        self.train_mask, self.val_mask, self.test_mask = train_mask, val_mask, test_mask
        self.has_scheduler = args.has_scheduler
        self.args =args
    def forward(self, x, edge_index):
        out = self.model(x, edge_index)
        return out
    def on_train_epoch_end(self):
        # Print every 10 epochs (epoch numbers are 0-indexed)
        if (self.current_epoch + 1) % 30 == 0:
            # Access last logged val_acc from self.trainer.logger_connector.metrics
            val_acc = self.trainer.logged_metrics.get("val_acc", None)
            if val_acc is not None:
                print(f"Epoch {self.current_epoch + 1}: val_acc = {val_acc:.4f}")
            else:
                print(f"Epoch {self.current_epoch + 1}: val_acc not logged yet.")

    def training_step(self, batch, batch_idx):
        x, edge_index, y = batch
        out = self.model(x, edge_index)

        train_loss = nn.functional.nll_loss(out[self.train_mask], y[self.train_mask].squeeze())
        self.log("train_loss", train_loss)

        # val_loss = nn.functional.nll_loss(out[self.val_mask], y[self.val_mask].squeeze())
        # self.log("val_loss", val_loss)

        y_pred = out.max(1)[1]
        train_acc = self.evaluate(y_pred=y_pred[self.train_mask], y_true=y[self.train_mask])
        self.log("train_acc", train_acc)
        # val_acc = self.evaluate(y_pred=y_pred[self.val_mask], y_true=y[self.val_mask])
        # self.log("val_acc", val_acc)

        return train_loss


    def evaluate(self, y_pred, y_true):
        if self.evaluator:
            acc = self.evaluator.eval({"y_true": y_true, "y_pred": y_pred.unsqueeze(1)})["acc"]
        else:
            acc = y_pred.eq(y_true.squeeze()).sum().item() / y_pred.shape[0]
        return acc

    def test_step(self, batch, batch_idx):
        x, edge_index, y = batch
        out = self.model(x, edge_index)

        y_pred = out.max(1)[1]
        test_acc = self.evaluate(y_pred=y_pred[self.test_mask], y_true=y[self.test_mask])
        self.log("test_acc", test_acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, edge_index, y = batch
        out = self.model(x, edge_index)
        val_loss = nn.functional.nll_loss(out[self.val_mask], y[self.val_mask].squeeze())
        self.log("val_loss", val_loss, prog_bar=True)

        y_pred = out.max(1)[1]
        val_acc = self.evaluate(y_pred=y_pred[self.val_mask], y_true=y[self.val_mask])
        self.log("val_acc", val_acc)
        # return val_acc

    def configure_optimizers(self):
        other_params = []
        for name, param in self.model.named_parameters():
            other_params.append(param)

        if hasattr(self.model, 'coefs'):  # parameter without weight_decay will typically change faster
            optimizer = torch.optim.Adam(
                [dict(params=self.model.reg_params, lr=args.lr, weight_decay=5e-4), dict(params=self.model.non_reg_params, lr=self.args.lr, weight_decay=0),
                 dict(params=self.model.coefs, lr=self.args.coeflr * args.lr, weight_decay=self.args.wd4coef), ],
            )
        elif hasattr(self.model, 'reg_params'):
            optimizer = torch.optim.Adam(
                [dict(params=self.model.reg_params, weight_decay=5e-4), dict(params=self.model.non_reg_params, weight_decay=0), ], lr=self.args.lr)
            try:
                self.model.edge_weight.requires_grad = False
            except:
                pass
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.l2)

        # optimizer = optim.AdamW([{'params': other_params, 'weight_decay': self.weight_decay}], lr=self.lr)
        # print(optimizer)

        if self.has_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=self.args.patience)
            print("Have scheduler")
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            }

            # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": self.args.monitor}

        return optimizer

import torch.nn.functional as F

class LightingFullBatchModelWrapper(pl.LightningModule):
    def __init__(self, model, args, train_mask, val_mask, test_mask, evaluator=None):
        super().__init__()
        self.model = model
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.evaluator = evaluator
        self.train_mask, self.val_mask, self.test_mask = train_mask, val_mask, test_mask

    def training_step(self, batch, batch_idx):
        x, edge_index, y = batch
        # x, y, edge_index = batch.x, batch.y.long(), batch.edge_index
        out = self.model(x, edge_index)

        train_loss = F.cross_entropy(out[self.train_mask], y[self.train_mask])
        self.log("train_loss", train_loss)

        val_loss = F.cross_entropy(out[self.val_mask], y[self.val_mask])
        self.log("val_loss", val_loss)

        y_pred = out.max(1)[1]
        train_acc = self.evaluate(y_pred=y_pred[self.train_mask], y_true=y[self.train_mask])
        self.log("train_acc", train_acc)
        val_acc = self.evaluate(y_pred=y_pred[self.val_mask], y_true=y[self.val_mask])
        self.log("val_acc", val_acc)

        return train_loss

    def on_train_epoch_end(self):
        # Print every 10 epochs (epoch numbers are 0-indexed)
        if (self.current_epoch + 1) % 30 == 0:
            # Access last logged val_acc from self.trainer.logger_connector.metrics
            val_acc = self.trainer.logged_metrics.get("val_acc", None)
            if val_acc is not None:
                print(f"Epoch {self.current_epoch + 1}: val_acc = {val_acc:.4f}")
            else:
                print(f"Epoch {self.current_epoch + 1}: val_acc not logged yet.")

    def evaluate(self, y_pred, y_true):
        if self.evaluator:
            acc = self.evaluator.eval({"y_true": y_true, "y_pred": y_pred.unsqueeze(1)})["acc"]
        else:
            acc = y_pred.eq(y_true.squeeze()).sum().item() / y_pred.shape[0]
        return acc

    def test_step(self, batch, batch_idx):
        x, y, edge_index = batch.x, batch.y.long(), batch.edge_index
        out = self.model(x, edge_index)

        y_pred = out.max(1)[1]
        test_acc = self.evaluate(y_pred=y_pred[self.test_mask], y_true=y[self.test_mask])
        self.log("test_acc", test_acc, prog_bar=True)
        print("test_acc", test_acc)

    def configure_optimizers(self):
        other_params= []

        for name, param in self.model.named_parameters():
            other_params.append(param)

        optimizer = optim.AdamW([{'params': other_params, 'weight_decay': self.weight_decay}], lr = self.lr)
        print(optimizer)
        return optimizer
