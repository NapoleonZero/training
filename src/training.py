import pkbar # progress bar for pytorch
import torch
from torch.utils.data import DataLoader
from datasets.utils import split_dataset

class TrainingLoop():
    def __init__(self, model, dataset, loss_fn, optimizer, train_p=0.7, val_p=0.15, test_p=0.15, batch_size=1024, device='cpu', verbose=1, seed=42):
        self.model = model
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_p = train_p
        self.val_p = val_p
        self.test_p = test_p
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose
        self.seed = 42

        # TODO: maybe use an other class for evaluation?
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self._make_dataloaders(
                self.dataset,
                self.train_p,
                self.val_p,
                self.test_p)

    def _make_dataloaders(self, dataset, train_p, val_p, test_p):
        train_ds, val_ds, test_ds = split_dataset(dataset,
                train_p, val_p, test_p,
                seed=self.seed)
        train_dl = DataLoader(train_ds,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=4,
                prefetch_factor=4)
        val_dl = DataLoader(val_ds,
                batch_size=len(val_ds),
                shuffle=False,
                pin_memory=True,
                num_workers=0)
        test_dl = DataLoader(test_ds,
                batch_size=len(test_ds),
                shuffle=False,
                pin_memory=True,
                num_workers=0)
        return train_dl, val_dl, test_dl


    def _train(self, epochs):
        num_batches = len(self.train_dataloader)
        self.model.train()
        for epoch in range(epochs):
            ################################### Initialization ########################################
            kbar = pkbar.Kbar(target=num_batches, epoch=epoch, num_epochs=epochs, width=30, always_stateful=False)
            # By default, all metrics are averaged over time. If you don't want this behavior, you could either:
            # 1. Set always_stateful to True, or
            # 2. Set stateful_metrics=["loss", "rmse", "val_loss", "val_rmse"], Metrics in this list will be displayed as-is.
            # All others will be averaged by the progbar before display.
            ###########################################################################################

            # running_loss = 0.0
            for batch, (X, y) in enumerate(self.train_dataloader):
                # TODO: generalize variable type
                X, y = X.float().to(self.device), y.float().to(self.device)

                # Compute prediction error
                pred = self.model(X).squeeze()
                loss = self.loss_fn(pred, y)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                kbar.update(batch, values=[("loss", loss.item() )])

            val_loss = self._test(self.val_dataloader)

            ################################ Add validation metrics ###################################
            kbar.add(1, values=[("val_loss", val_loss)])
            ###########################################################################################

    def _test(self, dataloader):
        num_batches = len(dataloader)
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X, y in dataloader:
                # TODO: generalize variable type
                X, y = X.float().to(self.device), y.float().to(self.device)
                pred = self.model(X).squeeze()
                test_loss += self.loss_fn(pred, y).item()
        test_loss /= num_batches
        return test_loss

    def run(self, epochs=10):
        self._train(epochs)
        return self.model

    # TODO: maybe use an other class for evaluation?
    def evaluate(self):
        pass
