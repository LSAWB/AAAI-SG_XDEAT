import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

class ModelTrainer:
    def __init__(self, model, optimizer, scheduler, device, no_of_classes):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.no_of_classes = no_of_classes

    def train(self, train_loader, epoch):
        
        self.model.train()
        train_loss = 0.0
        ground_truths, preds_logits = [], []

        for data in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            self.optimizer.zero_grad()

            cat_data, num_data, t_cat_data, labels = self._prepare_data(data)
            logits = self.model(cat_data, num_data, t_cat_data)
            labels = labels.squeeze().long()

            classification_loss = F.cross_entropy(logits, labels)
            total_loss = classification_loss

            total_loss.backward()
            train_loss += total_loss.item()

            self.optimizer.step()
            self.scheduler.step()

            preds_logits.append(logits.detach())
            ground_truths.append(labels.detach())

        return self._evaluate(ground_truths, preds_logits, train_loader, train_loss)


    def evaluate(self, val_loader, epoch):
        
        self.model.eval()
        valid_loss = 0.0
        ground_truths, preds_logits = [], []

        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Evaluating Epoch {epoch+1}", leave=False):

                cat_data, num_data, t_cat_data, labels = self._prepare_data(data)
                logits = self.model(cat_data, num_data, t_cat_data)
                labels = labels.squeeze().long()

                classification_loss = F.cross_entropy(logits, labels)
                total_loss = classification_loss

                valid_loss += total_loss.item()

                preds_logits.append(logits.detach())
                ground_truths.append(labels.detach())

        return self._evaluate(ground_truths, preds_logits, val_loader, valid_loss)  

    
    def test(self, test_loader, epoch):
        self.model.eval()
        test_loss = 0.0
        ground_truths, preds_logits = [], []

        with torch.no_grad():
            for data in tqdm(test_loader, desc=f"Testing Epoch {epoch+1}", leave=False):

                cat_data, num_data, t_cat_data, labels = self._prepare_data(data)
                logits = self.model(cat_data, num_data, t_cat_data)
                labels = labels.squeeze().long()
                
                classification_loss = F.cross_entropy(logits, labels)
                total_loss = classification_loss

                test_loss += total_loss.item()

                preds_logits.append(logits.detach())
                ground_truths.append(labels.detach())

        return self._evaluate(ground_truths, preds_logits, test_loader, test_loss)    


    def _prepare_data(self, data):
        cat_data, num_data, t_cat_data, labels = data
        return (
            cat_data.to(self.device, dtype=torch.long),
            num_data.to(self.device, dtype=torch.float32),
            t_cat_data.to(self.device, dtype=torch.float32),
            labels.to(self.device, dtype=torch.float32)
        )

    def _evaluate(self, ground_truths, preds_logits, loader, loss):
        ground_truths = torch.cat(ground_truths)
        preds_logits = torch.cat(preds_logits)
        loss /= len(loader)

        # Extract predicted labels
        preds_probs = torch.sigmoid(preds_logits)
        preds_labels = preds_probs.argmax(dim=1)

        # Convert tensors to numpy arrays
        y_true = ground_truths.cpu().numpy()
        y_pred = preds_labels.cpu().numpy()

        # Use sklearn to compute accuracy
        accuracy = accuracy_score(y_true, y_pred)

        return loss, accuracy