import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ImageClassificationModel(nn.Module):
    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = self.accuracy(out, labels)
        return loss, acc

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = self.accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        lr = result.get('lr', [0.0])
        train_loss = result.get('train_loss', 0.0)
        val_loss = result.get('val_loss', 0.0)
        val_acc = result.get('val_acc', 0.0)
        train_acc = result.get('train_acc', 0.0)

        if isinstance(lr, list):
            lr = lr[-1]

        print(f"Epoch [{epoch}], "
              f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
              f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def evaluate(self, val_loader):
        self.eval()
        outputs = [self.validation_step(batch) for batch in val_loader]
        return self.validation_epoch_end(outputs)

    def fit(self, epochs, max_lr, train_loader, val_loader, weight_decay=0, grad_clip=None, opt_func=optim.SGD):
        history = []
        optimizer = opt_func(self.parameters(), max_lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
        
        for epoch in range(epochs):
            self.train()
            train_losses = []
            train_accs = []
            lrs = []
            for batch in train_loader:
                loss, acc = self.training_step(batch)
                train_losses.append(loss)
                train_accs.append(acc)
                loss.backward()
                
                if grad_clip:
                    nn.utils.clip_grad_value_(self.parameters(), grad_clip)
                
                optimizer.step()
                optimizer.zero_grad()
                
                lrs.append(self.get_lr(optimizer))
                sched.step()
                
            result = self.evaluate(val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            result['train_acc'] = torch.stack(train_accs).mean().item()
            result['lr'] = lrs
            self.epoch_end(epoch, result)
            history.append(result)
                
        return history
