from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from IPython.display import clear_output
import copy
import os
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


class AbsVoxelNNInferrer(nn.Module):
    """
    Abstract base class for voxel inference neural networks.
    Provides training, evaluation, saving/loading, and utility functions.

    Subclasses are expected to define their own forward() method and architecture.
    """

    def __init__(self):
        super().__init__()

    def _collate_poca_batch(self, batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Collates a batch of (poca_tensor, x0_tensor) pairs for training or evaluation.

        Pads variable-length PoCA sequences in the batch
        to the same length, and generates corresponding masks to recover unpadded entries.

        Args:
            batch (List[Tuple[Tensor, Tensor]]):
                A list of tuples, where each tuple contains:
                - poca_tensor: (N_i, feature_dim) float tensor for i-th example
                - x0_tensor: ground truth flattened X0 tensor

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - padded_poca_tensor: (B, max_len, feature_dim) tensor with padded PoCA sequences
                - x0_tensor: (B, X, Y, Z) tensor of ground truth volumes
                - mask_tensor: (B, max_len) boolean tensor indicating unpadded PoCA entries
        """
        poca_batch = [b[0] for b in batch]  # list of (N_i, feature_dim)
        
        # reshape flattened X0 tensor to 3D. 
        # In TomOpt, X0 is originally shaped like Z,X,Y so need to permute
        # TomOpt X0 voxelization is of shape (4,10,10)
        x0_batch = [b[1].view(4,10,10).permute(1,2,0) for b in batch] 
        x0_tensor = torch.stack(x0_batch)
    
        # Pad sequences to the same length
        # batch_first=True -> output shape: (B, max_len, feature_dim)
        padded_poca = pad_sequence(poca_batch, batch_first=True, padding_value=0.0)
    
        # Create mask: True for real entries, False for padding
        lengths = torch.tensor([p.shape[0] for p in poca_batch])
        max_len = padded_poca.shape[1]
        mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        mask = mask.to(torch.bool)
    
        return padded_poca, x0_tensor, mask
    
    def _run_epoch(self, loader:DataLoader, optimizer, scheduler, loss_fn, device:str, epoch:int, train:bool=True) -> float:
        """
        Runs one epoch of training or validation.
        """
        self.train() if train else self.eval()
        total_loss = 0.0

        for poca_batch, x0_true, point_mask in tqdm(loader):
            poca_batch = poca_batch.to(device, non_blocking=True)
            x0_true = x0_true.to(device, non_blocking=True)
            x0_true = torch.log(x0_true)#.permute(0, 2, 3, 1)

            with torch.set_grad_enabled(train):
                x0_pred = self(poca_batch, point_mask)
                loss = loss_fn(x0_pred, x0_true)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _plot_training(self, train_losses, val_losses, lr_history) -> None:
        """
        Plots training/validation loss and learning rate.
        """
        epochs = list(range(1, len(train_losses) + 1))
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8),
                                       gridspec_kw={'height_ratios': [3, 1]})

        ax1.plot(epochs, train_losses, label='Train Loss')
        ax1.plot(epochs, val_losses, label='Val Loss')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, lr_history, label='Learning Rate', color='tab:orange')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('LR')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def train_model(self,
                    train_set,
                    val_set, 
                    n_epochs:int = 20, 
                    lr:float = 1e-4,
                    batch_size: int = 20,
                    live_plot=True, 
                    loss_fun: Callable = None, 
                    optim = None,
                    schedlr = None,
                    collate_fun = None,
                    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                    use_wandb: bool = False,
                    wandb_config:Dict = None,
                    **kwargs,
                   ):
        """
        High-level training loop.
        """
        self.to(device)
        collate_fn = collate_fun if collate_fun is not None else self._collate_poca_batch
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn= collate_fn, num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_set,  batch_size=batch_size, shuffle=False, collate_fn= collate_fn, num_workers=0, pin_memory=True)

        loss_fn = loss_fun if loss_fun is not None else nn.HuberLoss(delta=1.0)
        optimizer = optim if optim is not None else torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = schedlr if schedlr is not None else torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.3, patience=3, min_lr=1e-6, threshold_mode='rel', verbose=True
        )

        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(self.state_dict())
        train_losses, val_losses, lr_history = [], [], []
        
        #  optional Weights & Biases (WandB) logging 
        if use_wandb:
            try:
                import wandb
                if wandb_config is None:
                    wandb_config = {
                        "learning_rate": lr,
                        "batch_size": batch_size,
                        "epochs": n_epochs,
                        "loss": "Huber",
                        "optimizer": "AdamW"
                    }
                wandb.init(project="voxel-x0", config=wandb_config)
                wandb.watch(self, log="gradients", log_freq=10)
            except ImportError:
                print("wandb not installed; proceeding without logging.")
                use_wandb = False

        for epoch in range(n_epochs):
            avg_train = self._run_epoch(train_loader, optimizer, scheduler, loss_fn, device, epoch, train=True)
            avg_val   = self._run_epoch(val_loader, None, None, loss_fn, device, epoch, train=False)

            train_losses.append(avg_train)
            val_losses.append(avg_val)
            lr_history.append(scheduler.get_last_lr()[0])
            scheduler.step(avg_val)

            print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | LR: {lr_history[-1]:.2e}")

            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train,
                    "val_loss": avg_val,
                    "lr": lr_history[-1]
                })

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_model_wts = copy.deepcopy(self.state_dict())

            if live_plot:
                clear_output(wait=True)
                self._plot_training(train_losses, val_losses, lr_history)

        self.load_state_dict(best_model_wts)

        if use_wandb:
            wandb.finish()

        return train_losses, val_losses, lr_history

    def save_model(self, path: str):
        """
        Saves the model's state_dict to the specified path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Model saved to: {path}")

    def load_model(self, path: str, map_location: str = 'cpu'):
        """
        Loads the model's weights from a saved state_dict.
        """
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)
        print(f"Model loaded from: {path}")

    def evaluate_on_dataset(self, dataset, batch_size=20, device='cuda') -> float:
        """
        Evaluates the model on a given dataset and returns average loss.
        """
        loader = self._get_dataloader(dataset, batch_size, shuffle=False)
        self.to(device)
        self.eval()

        loss_fn = nn.HuberLoss(delta=1.0)
        total_loss = 0.0
        with torch.no_grad():
            for poca_batch, x0_true, point_mask in loader:
                poca_batch = poca_batch.to(device)
                x0_true = x0_true.to(device)
                x0_true = torch.log(x0_true)#.permute(0, 2, 3, 1)
                x0_pred = self(poca_batch, point_mask)
                total_loss += loss_fn(x0_pred, x0_true)

        avg_loss = total_loss / len(loader)
        print(f"Evaluation Loss: {avg_loss:.4f}")
        return avg_loss

    def count_parameters(self):
        """
        Returns the number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
