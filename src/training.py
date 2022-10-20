import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from datasets.utils import split_dataset
from contextlib import ExitStack
import numpy as np
import gc

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
                 filter_fn=None,
                 device='cpu',
                 mixed_precision=False,
                 callbacks=[],
                 val_metrics={},
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
        self.filter_fn = filter_fn
        self.device = device
        self.mixed_precision = mixed_precision
        self.verbose = verbose
        self.callbacks = callbacks
        self.val_metrics = val_metrics
        self.seed = 42

        self._clear_state()
        self._init_dataloaders()

    def _clear_state(self):
        """ Clear internal training state """
        self.state = {}
        self.metrics = {}

    def _init_dataloaders(self):
        # TODO: maybe use an other class for evaluation?
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self._make_dataloaders(
                self.dataset,
                self.train_p,
                self.val_p,
                self.test_p)

    def _collate_fn(self, batch):
        if self.filter_fn:
            batch = [data for data in batch if self.filter_fn(data)]
        return torch.utils.data.default_collate(batch)

    def _make_dataloaders(self, dataset, train_p, val_p, test_p):
        train_ds, val_ds, test_ds = split_dataset(dataset,
                train_p, val_p, test_p,
                seed=self.seed)
        train_dl = DataLoader(train_ds,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                collate_fn=self._collate_fn,
                pin_memory=True,
                num_workers=8,
                prefetch_factor=8,
                worker_init_fn=dataset.worker_init_fn,
                persistent_workers=False)
        val_dl = DataLoader(val_ds,
                batch_size=self.batch_size,
                collate_fn=self._collate_fn,
                shuffle=False,
                pin_memory=True,
                num_workers=2,
                prefetch_factor=8,
                persistent_workers=False)
        test_dl = DataLoader(test_ds,
                batch_size=self.batch_size,
                collate_fn=self._collate_fn,
                shuffle=False,
                pin_memory=True,
                num_workers=0)
        return train_dl, val_dl, test_dl

    def _train(self, epochs):
        if self.train_dataloader is None or self.val_dataloader is None:
            self._init_dataloaders()

        num_batches = len(self.train_dataloader)
        self.update_state('batches', num_batches)
        
        self.on_train_start()

        # Pytorch auto scaler for mixed precision training
        if self.mixed_precision:
            scaler = torch.cuda.amp.GradScaler()

        initial_epoch = self.get_state('epoch', 1)

        #######################################################################
        # TESTING AUX LOSS
        # aux_loss = nn.MSELoss()
        #######################################################################

        for epoch in range(initial_epoch, initial_epoch + epochs):
            self.on_train_epoch_start(epoch)

            self.model.train()
            for batch, (X, aux, y) in enumerate(self.train_dataloader):
                self.on_train_batch_start(batch)
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
        #######################################################################
                    # TESTING AUX LOSS
                    # aux_pred = self.model.material_mlp(X).squeeze()
                    # loss = self.loss_fn(pred, y) + aux_loss(aux_pred, y)
                    loss = self.loss_fn(pred, y)
        #######################################################################

                # Backpropagation
                if self.mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                self.on_train_batch_end(batch, loss.detach())

            val_loss, other_metrics = self._test(self.val_dataloader, self.val_metrics)
            self.on_validation_end(val_loss, other_metrics)
            self.on_train_epoch_end(epoch)

        self.on_train_end()

    def _test(self, dataloader, metrics):
        num_batches = len(dataloader)
        self.model.eval()
        test_loss = 0.0
        test_metrics = dict.fromkeys(metrics.keys(), 0.0)
        # test_metrics = [0.0 for _ in metrics.keys()]
        with torch.no_grad():
            for X, aux, y in dataloader:
                # TODO: generalize variable type
                X = X.float().to(self.device, non_blocking=True, memory_format=torch.channels_last)
                y = y.float().to(self.device, non_blocking=True)
                aux = aux.float().to(self.device, non_blocking=True)

                pred = self.model(X, aux).squeeze().detach()
                test_loss += self.loss_fn(pred, y).detach()
                for name, fn in metrics.items():
                    test_metrics[name] += fn(pred, y).detach()

        test_loss /= num_batches
        test_metrics = {k: v / num_batches for k, v in test_metrics.items()}
        return test_loss, test_metrics

    def predict(self, data):
        """ Run inference over a set of datapoints,
            returning the predicted values.
            `data` can be any iterable complying with the model input.
        """
        self.model.eval()
        h = []
        with torch.no_grad():
            for X, aux in data:
                X = X.float().to(self.device, non_blocking=True, memory_format=torch.channels_last)
                aux = aux.float().to(self.device, non_blocking=True)

                pred = self.model(X, aux).squeeze()
                h = np.concatenate((h, pred.cpu().detach().numpy()), axis=None)

        return h
    
    def run(self, epochs=10):
        try:
            self._train(epochs)
        except KeyboardInterrupt:
            print('Terminating training loop...')

        return self.model

    def clear(self):
        del self.train_dataloader
        del self.val_dataloader
        del self.test_dataloader
        self._clear_state()
    
    def dump_state(self):
        """ Produces a snapshot dictionary containing all information
            to restore the current state of the TrainingLoop sometime in the
            future.
        """
        return {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_loop_state_dict': self.get_states(),
                'training_loop_metrics': self.get_last_metrics(),
                'callbacks_state_dict': [c.state_dict() for c in self.callbacks],
                # 'train_dataloader': self.train_dataloader,
                # 'val_dataloader': self.val_dataloader,
                # 'test_dataloader': self.test_dataloader
                }

    def load_state(self, dump):
        """ Loads a TrainingLoop snapshot produced by a call to dump_state. """
        self.model.load_state_dict(dump['model_state_dict'])
        self.optimizer.load_state_dict(dump['optimizer_state_dict'])
        self.state = dump['training_loop_state_dict']
        self.metrics = dump['training_loop_metrics']

        for i, d in enumerate(dump['callbacks_state_dict']):
            self.callbacks[i].load_state_dict(d)

    def get_last_metric(self, metric, default=None):
        """ Get last computed metric """
        return self.metrics.get(metric, default)

    def get_last_metrics(self):
        return self.metrics.copy()

    def update_metric(self, metric, value):
        """ Update the given metric with the current value
            The method automatically tracks min and max values of the metric
        """
        min_v = self.get_last_metric(f'min-{metric}')
        max_v = self.get_last_metric(f'max-{metric}')
        self.metrics[metric] = value
        if min_v is None or value < min_v:
            self.metrics[f'min-{metric}'] = value
        if max_v is None or value > max_v:
            self.metrics[f'max-{metric}'] = value

    def update_state(self, key, value):
        self.state[key] = value

    def get_state(self, key, default=None):
        return self.state.get(key, default)

    def get_states(self):
        return self.state.copy()

    """ Callback hooks """
    def on_train_start(self):
        for c in self.callbacks: c.on_train_start(self)

    def on_train_end(self):
        for c in self.callbacks: c.on_train_end(self)

    def on_train_batch_start(self, batch_num):
        self.update_state('batch', batch_num)
        for c in self.callbacks: c.on_train_batch_start(self)

    def on_train_batch_end(self, batch_num, batch_loss):
        # Update current batch loss
        self.update_metric('loss', batch_loss)
        # Current mean loss (default 0 if None)
        mean_loss = self.get_last_metric('mean_loss', 0.0)
        # Running mean loss update
        mean_loss = mean_loss + (batch_loss - mean_loss)/(batch_num + 1)
        self.update_metric('mean_loss', mean_loss)
        for c in self.callbacks: c.on_train_batch_end(self)

        # if batch_num % 10000 == 0:
        #     torch.cuda.empty_cache()
        #     gc.collect()

    def on_train_epoch_start(self, epoch_num):
        self.update_metric('mean_loss', 0.0)
        self.update_state('epoch', epoch_num)
        for c in self.callbacks: c.on_train_epoch_start(self)

    def on_train_epoch_end(self, epoch_num):
        # setting epoch + 1 on epoch end is necessary for cases in which
        # the training stopped mid-epoch
        self.update_state('epoch', epoch_num + 1)
        for c in self.callbacks: c.on_train_epoch_end(self)
        torch.cuda.empty_cache()

    def on_validation_batch_start(self):
        for c in self.callbacks: c.on_validation_batch_start(self)

    def on_validation_batch_end(self):
        for c in self.callbacks: c.on_validation_batch_end(self)

    def on_validation_start(self):
        for c in self.callbacks: c.on_validation_start(self)

    def on_validation_end(self, val_loss, other_metrics):
        self.update_metric('val_loss', val_loss)

        for metric, value in other_metrics.items():
            self.update_metric(f'val_{metric}', value)
        for c in self.callbacks: c.on_validation_end(self)
        torch.cuda.empty_cache()

    # TODO: maybe use an other class for evaluation?
    def evaluate(self):
        loss, other_metrics = self._test(self.test_dataloader, self.val_metrics)
        return {'loss': loss, **other_metrics}
