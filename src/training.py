import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from datasets.utils import split_dataset
from contextlib import ExitStack

class TrainingLoop():
    def __init__(self,
                 model,
                 dataset,
                 loss_fn,
                 optimizer,
                 train_p=0.7,
                 val_p=0.15,
                 test_p=0.15,
                 batch_size=1024,
                 shuffle=False,
                 device='cpu',
                 mixed_precision=False,
                 callbacks=[],
                 metrics=[],
                 verbose=1,
                 seed=42):
        self.model = model
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_p = train_p
        self.val_p = val_p
        self.test_p = test_p
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.mixed_precision = mixed_precision
        self.verbose = verbose
        self.callbacks = callbacks
        self.metrics = dict.fromkeys(metrics)
        self.seed = 42

        # Internal training state
        self.state = {}

    def _clear_state(self):
        """ Clear internal training state """
        self.state['epoch'] = None
        self.state['lr'] = None

    def _init_dataloaders(self):
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
                shuffle=self.shuffle,
                pin_memory=True,
                num_workers=4,
                prefetch_factor=2,
                persistent_workers=True)
        val_dl = DataLoader(val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=4,
                prefetch_factor=2,
                persistent_workers=True)
        test_dl = DataLoader(test_ds,
                batch_size=len(test_ds),
                shuffle=False,
                pin_memory=True,
                num_workers=0)
        return train_dl, val_dl, test_dl

    def _train(self, epochs):
        self._clear_state()
        self._init_dataloaders()
        num_batches = len(self.train_dataloader)
        self.update_state('batches', num_batches)
        
        self.on_train_start()

        # Pytorch auto scaler for mixed precision training
        if self.mixed_precision:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(epochs):
            self.update_state('epoch', epoch)
            self.on_train_epoch_start()

            self.model.train()
            for batch, (X, aux, y) in enumerate(self.train_dataloader):
                self.update_state('batch', batch)
                # TODO: generalize variable type
                X = X.float().to(self.device, non_blocking=True, memory_format=torch.channels_last)
                y = y.float().to(self.device, non_blocking=True)
                aux = aux.float().to(self.device, non_blocking=True)

                # Clear gradients
                self.optimizer.zero_grad(set_to_none=True)

                # Forward pass
                with ExitStack() as stack:
                    if self.mixed_precision:
                        stack.enter_context(torch.cuda.amp.autocast())
                    pred = self.model(X, aux).squeeze()
                    loss = self.loss_fn(pred, y)

                # Backpropagation
                if self.mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                self.update_metric('loss', loss)
                self.on_train_batch_end()

            val_loss = self._test(self.val_dataloader)
            self.update_metric('val_loss', val_loss)
            self.on_validation_end()

        self.on_train_end()

    def _test(self, dataloader):
        num_batches = len(dataloader)
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X, aux, y in dataloader:
                # TODO: generalize variable type
                X = X.float().to(self.device, non_blocking=True, memory_format=torch.channels_last)
                y = y.float().to(self.device, non_blocking=True)
                aux = aux.float().to(self.device, non_blocking=True)

                pred = self.model(X, aux).squeeze()
                test_loss += self.loss_fn(pred, y)
        test_loss /= num_batches
        return test_loss

    def run(self, epochs=10):
        self._train(epochs)
        return self.model

    def clear(self):
        del self.train_dataloader
        del self.val_dataloader
        del self.test_dataloader
        self._clear_stae()

    def get_last_metric(self, metric):
        """ Get last computed metric """
        return self.metrics.get(metric, None)

    def update_metric(self, metric, value):
        self.metrics[metric] = value

    def update_state(self, key, value):
        self.state[key] = value

    def get_state(self, key):
        return self.state.get(key, None)

    """ Callback hooks """
    def on_train_start(self):
        for c in self.callbacks: c.on_train_start(self)

    def on_train_end(self):
        for c in self.callbacks: c.on_train_end(self)

    def on_train_batch_start(self):
        for c in self.callbacks: c.on_train_batch_start(self)

    def on_train_batch_end(self):
        for c in self.callbacks: c.on_train_batch_end(self)

    def on_train_epoch_start(self):
        for c in self.callbacks: c.on_train_epoch_start(self)

    def on_train_epoch_end(self):
        for c in self.callbacks: c.on_train_epoch_end(self)

    def on_validation_batch_start(self):
        for c in self.callbacks: c.on_validation_batch_start(self)

    def on_validation_batch_end(self):
        for c in self.callbacks: c.on_validation_batch_end(self)

    def on_validation_start(self):
        for c in self.callbacks: c.on_validation_start(self)

    def on_validation_end(self):
        for c in self.callbacks: c.on_validation_end(self)

    # TODO: maybe use an other class for evaluation?
    def evaluate(self):
        pass
