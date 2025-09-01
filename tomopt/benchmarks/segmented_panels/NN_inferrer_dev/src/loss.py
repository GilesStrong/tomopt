import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaterialReconstructionLoss(nn.Module):
    """
    Comprehensive loss for material block reconstruction that addresses:
    1. Sparsity of material blocks in mostly-air volumes
    2. Sharp material boundaries
    3. Different material properties
    """
    def __init__(self, 
                 focal_alpha=0.25, 
                 focal_gamma=2.0,
                 dice_weight=0.3,
                 boundary_weight=0.2,
                 consistency_weight=0.1):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.consistency_weight = consistency_weight
        
    def focal_loss(self, pred, target):
        """
        Focal loss to handle extreme class imbalance (lots of air, few material voxels)
        """
        # Convert to probabilities if needed
        if pred.max() > 1.0 or pred.min() < 0.0:
            pred = torch.sigmoid(pred)
            
        # Binary classification: material vs air 
        # Assume target > threshold means material
        material_threshold = target.mean() + target.std()  # Dynamic threshold
        target_binary = (target < material_threshold).float()
        
        # Focal loss computation
        ce_loss = F.binary_cross_entropy(pred, target_binary, reduction='none')
        pt = torch.where(target_binary == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.focal_gamma
        
        if self.focal_alpha is not None:
            alpha_weight = torch.where(target_binary == 1, self.focal_alpha, 1 - self.focal_alpha)
            focal_weight *= alpha_weight
            
        return (focal_weight * ce_loss).mean()
    
    def dice_loss(self, pred, target):
        """
        Dice loss for better handling of material regions
        """
        # Binarize predictions and targets
        material_threshold = target.mean() + 0.5 * target.std()
        pred_binary = (pred < material_threshold).float()
        target_binary = (target < material_threshold).float()
        
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum()
        
        dice = (2.0 * intersection) / (union + 1e-8)
        return 1 - dice
    
    def boundary_loss(self, pred, target):
        """
        Preserve sharp boundaries between materials and air
        """
        # Compute gradients in 3D
        def compute_gradients_3d(tensor):
            # tensor shape: (B, Z, X, Y)
            grad_z = torch.abs(tensor[:, 1:, :, :] - tensor[:, :-1, :, :])
            grad_x = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :])
            grad_y = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1])
            return grad_z, grad_x, grad_y
        
        pred_grad_z, pred_grad_x, pred_grad_y = compute_gradients_3d(pred)
        target_grad_z, target_grad_x, target_grad_y = compute_gradients_3d(target)
        
        # L1 loss on gradients to preserve edges
        boundary_loss = (
            F.l1_loss(pred_grad_z, target_grad_z) +
            F.l1_loss(pred_grad_x, target_grad_x) + 
            F.l1_loss(pred_grad_y, target_grad_y)
        ) / 3.0
        
        return boundary_loss
    
    def consistency_loss(self, pred, target):
        """
        Ensure spatial consistency - nearby voxels should have similar values
        """
        # 3D smoothness loss
        def smoothness_loss_3d(tensor):
            diff_z = torch.abs(tensor[:, 1:, :, :] - tensor[:, :-1, :, :])
            diff_x = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :])
            diff_y = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1])
            return (diff_z.mean() + diff_x.mean() + diff_y.mean()) / 3.0
        
        pred_smooth = smoothness_loss_3d(pred)
        target_smooth = smoothness_loss_3d(target)
        
        return F.l1_loss(pred_smooth, target_smooth)
    
    def forward(self, pred, target):
        # Base regression loss
        mse_loss = F.mse_loss(pred, target)
        
        # Specialized losses
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        consistency = self.consistency_loss(pred, target)
        
        total_loss = (
            mse_loss + 
            focal +
            self.dice_weight * dice +
            self.boundary_weight * boundary +
            self.consistency_weight * consistency
        )
        
        return total_loss.mean()


class MultiMaterialLoss(nn.Module):
    """
    For scenarios with multiple distinct materials (not just air vs material)
    pred, target: (B, Z, X, Y) where B is batch size
    """
    def __init__(self, num_materials=6, material_weights=None):
        super().__init__()
        self.num_materials = num_materials
        self.material_weights = material_weights or torch.ones(num_materials)
        
    def forward(self, pred, target):
        B, Z, X, Y = pred.shape
        
        # Convert continuous predictions to material probabilities
        # Expand pred to (B, num_materials, Z, X, Y)
        material_logits = pred.unsqueeze(1).repeat(1, self.num_materials, 1, 1, 1)
        
        # Create material templates (you might need to adjust these based on your data)
        material_templates = torch.linspace(0, 1, self.num_materials).to(pred.device)
        material_templates = material_templates.view(1, -1, 1, 1, 1)
        
        # Compute similarity to each material
        similarities = -torch.abs(material_logits - material_templates)
        similarities = similarities.view(B, self.num_materials, -1)  # (B, num_materials, Z*X*Y)
        
        # Convert target to material IDs
        target_materials = torch.round(target * (self.num_materials - 1)).long()
        target_materials = target_materials.clamp(0, self.num_materials - 1)
        target_materials = target_materials.reshape(B, -1)  # (B, Z*X*Y)
        
        # Compute cross entropy for each sample in batch
        total_loss = 0
        for b in range(B):
            ce_loss = F.cross_entropy(
                similarities[b].t(),  # (Z*X*Y, num_materials)
                target_materials[b],  # (Z*X*Y,)
                weight=self.material_weights.to(pred.device)
            )
            total_loss += ce_loss
        
        return total_loss / B


class ContrastiveMaterialLoss(nn.Module):
    """
    Contrastive loss to better separate different materials
    """
    def __init__(self, margin=1.0, temperature=0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        
    def forward(self, pred, target):
        # Flatten spatial dimensions
        pred_flat = pred.view(pred.size(0), -1)  # (B, H*W*D)
        target_flat = target.view(target.size(0), -1)
        
        # Compute pairwise similarities
        pred_norm = F.normalize(pred_flat, dim=1)
        similarities = torch.matmul(pred_norm, pred_norm.t()) / self.temperature
        
        # Create target similarity matrix
        target_norm = F.normalize(target_flat, dim=1)
        target_similarities = torch.matmul(target_norm, target_norm.t())
        
        # Contrastive loss: similar targets should have similar predictions
        positive_mask = target_similarities > 0.8  # Adjust threshold
        negative_mask = target_similarities < 0.2
        
        positive_loss = -similarities[positive_mask].mean()
        negative_loss = F.relu(self.margin - similarities[negative_mask]).mean()
        
        return positive_loss + negative_loss


class TVRegularizationLoss(nn.Module):
    """
    Total Variation loss for smoother reconstructions while preserving edges
    """
    def __init__(self, weight=1e-4):
        super().__init__()
        self.weight = weight
        
    def forward(self, pred):
        # 3D Total variation
        diff_z = torch.abs(pred[:, 1:, :, :] - pred[:, :-1, :, :])
        diff_x = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        diff_y = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        
        tv_loss = self.weight * (diff_z.mean() + diff_x.mean() + diff_y.mean())
        return tv_loss


class AdaptiveMaterialLoss(nn.Module):
    """
    Adaptive loss that adjusts weights based on training progress
    """
    def __init__(self):
        super().__init__()
        self.epoch = 0
        self.material_loss = MaterialReconstructionLoss()
        self.tv_loss = TVRegularizationLoss()
        
    def update_epoch(self, epoch):
        self.epoch = epoch
        
    def forward(self, pred, target):
        # Start with more regularization, gradually focus on accuracy
        reg_weight = max(0.1, 1.0 - self.epoch / 50.0)  # Decrease over 50 epochs
        
        main_loss, loss_dict = self.material_loss(pred, target)
        tv_loss = self.tv_loss(pred)
        
        total_loss = main_loss + reg_weight * tv_loss
        loss_dict['tv'] = tv_loss.item()
        loss_dict['reg_weight'] = reg_weight
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


# Usage example with your training loop
def get_material_loss_function(loss_type='comprehensive'):
    """
    Factory function to get the appropriate loss function
    """
    if loss_type == 'comprehensive':
        return MaterialReconstructionLoss(
            focal_alpha=0.25,
            focal_gamma=2.0,
            dice_weight=0.4,
            boundary_weight=0.3,
            consistency_weight=0.1
        )
    elif loss_type == 'multi_material':
        return MultiMaterialLoss(num_materials=6)
    elif loss_type == 'contrastive':
        return ContrastiveMaterialLoss(margin=1.0, temperature=0.1)
    elif loss_type == 'adaptive':
        return AdaptiveMaterialLoss()
    else:
        return MaterialReconstructionLoss()


# Modified training function to handle complex loss
def train_with_material_loss(model, train_loader, val_loader, epochs=100, lr=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Use comprehensive material loss
    criterion = get_material_loss_function('comprehensive')
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for batch_idx, (poca_data, voxel_target, point_mask) in enumerate(train_loader):
            poca_data = poca_data.to(model.device)
            voxel_target = voxel_target.to(model.device)
            
            # Remove log transform from parent class if using these losses
            # voxel_target = torch.log(voxel_target).permute(0, 2, 3, 1)
            
            optimizer.zero_grad()
            
            pred = model(poca_data, point_mask)
            loss, loss_dict = criterion(pred, voxel_target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}:')
                for key, val in loss_dict.items():
                    print(f'  {key}: {val:.6f}')
        
        scheduler.step()
        
        # Update adaptive loss if using it
        if hasattr(criterion, 'update_epoch'):
            criterion.update_epoch(epoch)
            
        avg_loss = np.mean(epoch_losses)
        print(f'Epoch {epoch}: Average Loss: {avg_loss:.6f}')
    
    return model



class MaterialAwareLoss(nn.Module):
    """
    Loss function that weights predictions based on material difficulty.
    Gives higher weight to lighter materials that are harder to detect.
    """
    def __init__(self, 
                 material_weights=None,
                 base_weight=1.0,
                 light_material_boost=3.0,
                 focal_alpha=0.25, 
                 focal_gamma=2.0,
                 dice_weight=0.3,
                 boundary_weight=0.2):
        super().__init__()
        
        # Define material difficulty weights based on atomic number / scattering strength
        if material_weights is None:
            self.material_weights = {
                'uranium': 1.0,      # Easiest to detect
                'steel': 2.0,        
                'iron': 2.5,         
                'aluminum': 4.0,     # Harder to detect
                'beryllium': 5.0     # Hardest to detect
            }
        else:
            self.material_weights = material_weights
            
        self.base_weight = base_weight
        self.light_material_boost = light_material_boost
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
    
    def get_material_weight_map(self, target, material_type=None):
        """
        Create a weight map based on target values.
        Assumes lower X0 values correspond to denser materials.
        """
        if material_type is not None and material_type in self.material_weights:
            # If we know the material type, use direct weighting
            weight = self.material_weights[material_type]
            return torch.full_like(target, weight)
        else:
            # Infer material density from X0 values
            # Higher X0 -> lighter material -> higher weight
            normalized_target = (target - target.min()) / (target.max() - target.min() + 1e-8)
            
            # Create weight map: lighter materials (higher X0) get higher weights
            weight_map = self.base_weight + self.light_material_boost * normalized_target
            
            # Focus weights on material regions (non-air voxels)
            material_threshold = target.mean() + 0.5 * target.std()
            material_mask = target < material_threshold
            weight_map = torch.where(material_mask, weight_map, torch.ones_like(weight_map))
            
            return weight_map
    
    def weighted_mse_loss(self, pred, target, weight_map):
        """MSE loss with material-aware weighting"""
        mse = (pred - target) ** 2
        weighted_mse = mse * weight_map
        return weighted_mse.mean()
    
    def weighted_focal_loss(self, pred, target, weight_map):
        """Focal loss with material weighting"""
        # Convert to probabilities
        if pred.max() > 1.0 or pred.min() < 0.0:
            pred = torch.sigmoid(pred)
            
        # Binary classification: material vs air 
        material_threshold = target.mean() + target.std()
        target_binary = (target < material_threshold).float()
        
        # Focal loss computation
        ce_loss = F.binary_cross_entropy(pred, target_binary, reduction='none')
        pt = torch.where(target_binary == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.focal_gamma
        
        if self.focal_alpha is not None:
            alpha_weight = torch.where(target_binary == 1, self.focal_alpha, 1 - self.focal_alpha)
            focal_weight *= alpha_weight
        
        # Apply material weighting
        weighted_focal = focal_weight * ce_loss * weight_map
        return weighted_focal.mean()
    
    def boundary_loss(self, pred, target, weight_map):
        """Boundary loss with material weighting"""
        def compute_gradients_3d(tensor):
            grad_z = torch.abs(tensor[:, 1:, :, :] - tensor[:, :-1, :, :])
            grad_x = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :])
            grad_y = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1])
            return grad_z, grad_x, grad_y
        
        pred_grad_z, pred_grad_x, pred_grad_y = compute_gradients_3d(pred)
        target_grad_z, target_grad_x, target_grad_y = compute_gradients_3d(target)
        
        # Apply weights to gradients (reduce weight map dimensions accordingly)
        weight_z = weight_map[:, 1:, :, :]
        weight_x = weight_map[:, :, 1:, :]
        weight_y = weight_map[:, :, :, 1:]
        
        boundary_loss = (
            (torch.abs(pred_grad_z - target_grad_z) * weight_z).mean() +
            (torch.abs(pred_grad_x - target_grad_x) * weight_x).mean() + 
            (torch.abs(pred_grad_y - target_grad_y) * weight_y).mean()
        ) / 3.0
        
        return boundary_loss
    
    def forward(self, pred, target, material_type=None):
        # Get material-aware weight map
        weight_map = self.get_material_weight_map(target, material_type)
        
        # Weighted losses
        mse_loss = self.weighted_mse_loss(pred, target, weight_map)
        focal_loss = self.weighted_focal_loss(pred, target, weight_map)
        boundary_loss = self.boundary_loss(pred, target, weight_map)
        
        total_loss = (
            mse_loss + 
            focal_loss +
            self.boundary_weight * boundary_loss
        )
        
        return total_loss
        
def relative_mse(pred, target, weight_map=None, eps=1e-8):
    rel_err = ((pred - target) / (target + eps)) ** 2
    if weight_map is not None:
        rel_err = rel_err * weight_map
    return rel_err.mean()


class CurriculumLoss(nn.Module):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs

    def forward(self, pred, target, epoch):
        # Normalize (log scale assumed)
        eps = 1e-8
        rel_mse = (((pred - target) ** 2) / (target ** 2 + eps)).mean()

        # Create border mask
    
        # Extract material block region (excluding borders)
        border_width= 1
        depth, height, width = target.shape[1:]
        min_dim = min(depth, height, width)
        effective_border_width = min(border_width, min_dim // 4)
    
        interior_mask = torch.zeros_like(target, dtype=bool) 
        interior_mask[:,
        effective_border_width:-effective_border_width if effective_border_width > 0 else None,
        effective_border_width:-effective_border_width if effective_border_width > 0 else None,
        effective_border_width:-effective_border_width if effective_border_width > 0 else None] = True

        # Region masks
        mean, std = target.mean(), target.std()
        steel_mask = (target < mean - 0.5 * std).float()
        block_mask = (target >= mean - 0.5 * std) & interior_mask

        # Block flatness term: variance within block should be low
        block_pred = pred * block_mask
        block_flatness = block_pred.var()

        # Contrastive term: block mean far from steel mean
        block_mean = block_pred[block_mask > 0].mean()
        steel_mean = (pred * steel_mask)[steel_mask > 0].mean()
        contrastive = torch.relu(1.0 - (block_mean - steel_mean).abs())

        # Curriculum weights (linear schedule)
        progress = epoch / self.total_epochs
        w_rel = 1.0 - 0.5 * progress      # decrease over time
        w_flat = 0.5 * progress           # increase
        w_contrast = 0.5 * progress       # increase

        total_loss = ( 
            w_rel * rel_mse +
            w_flat * block_flatness +
            w_contrast * contrastive
        )
        return total_loss

