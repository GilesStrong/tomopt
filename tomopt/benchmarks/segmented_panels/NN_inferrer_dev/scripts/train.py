import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, "..")))
from typing import Tuple
import torch
import argparse
import json
from src.params import ExperimentParams
from src.data_h5 import MuonDataManager
from models.model_v0 import LocalAggVoxelX0InferNet
from src.loss import get_material_loss_function
def main():
    parser = argparse.ArgumentParser(description="Train MLP Voxel Inferrer Model")
    parser.add_argument("--config", required=True, type=str, help="Path to config JSON")
    parser.add_argument("--data", required=True, type=str, help="Path to HDF5 dataset")
    args = parser.parse_args()
    # Create output directory based on config name
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    out_dir = os.path.join("runs", config_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Outputs will be saved to: {out_dir}")
    # 1. Load Parameters
    print(f"Loading parameters from {args.config}...")
    params = ExperimentParams.from_json(args.config)
    
    device = torch.device(params.model.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 2. Setup Data
    print(f"Setting up data manager with {args.data}...")
    data_manager = MuonDataManager(hdf5_path=args.data)
    data_manager.load_all_data() # Loads HDF5 scans into RAM cache
    print("Splitting datasets based on Configuration...")
    train_indices, val_indices, test_indices = data_manager.create_train_val_test_split(
        n_positions=params.data.n_positions,
        n_materials=params.data.n_materials,
        n_repetitions=params.data.n_repetitions,
        val_combos=params.data.val_combos,
        test_combos=params.data.test_combos,
        random_seed=params.data.random_seed
    )
    # We use in_memory=True to leverage the RAM cache
    train_dataset, val_dataset, test_dataset = data_manager.create_datasets(
        train_indices, val_indices, test_indices, in_memory=True
    )

    # 3. Setup Model
    print("Initialize Model...")
    
    def _get_voxel_centers(voxel_shape = (10,10, 4), voxel_size: float = 0.1, origin: Tuple[float, float, float]=(0.0, 0.0, 0.3)):
        """
        Returns: (N_voxels, 3) tensor of voxel center coordinates
        """
        nx, ny, nz = voxel_shape
        ox, oy, oz = origin
    
        # Create grid indices
        x = torch.arange(nx)
        y = torch.arange(ny)
        z = torch.arange(nz)
    
        # Convert to physical coordinates (center of each voxel)
        xv, yv, zv = torch.meshgrid(x, y, z, indexing='ij')
        centers = torch.stack([xv, yv, zv], dim=-1).float()  # (nx, ny, nz, 3)
        centers = centers * voxel_size + voxel_size / 2.0
    
        # Shift by origin if provided
        centers[..., 0] += ox
        centers[..., 1] += oy
        centers[..., 2] += oz
    
        return centers.view(-1, 3)  # (N_voxels, 3)


    # need to specify origin as an argument, otherwise (0, 0, 0.3)
    
    voxel_centers =_get_voxel_centers();

    
    model = LocalAggVoxelX0InferNet(
        voxel_centers=voxel_centers,
        voxel_shape=params.model.voxel_shape,
        radius=params.model.radius,
        device=device
    )
    # 4. Determine Loss Function
    loss_str = params.train.loss_type
    print(f"Loading loss function: {loss_str}")
    if loss_str.lower() == "huber":
        criterion = torch.nn.HuberLoss(delta=1.0)
    else:
        # Load custom loss from src.loss
        criterion = get_material_loss_function(loss_str)
    # 5. Train
    print("Beginning Training...")
    train_losses, val_losses, lr_history = model.train_model(
        train_set=train_dataset,
        val_set=val_dataset,
        params=params,
        loss_fun=criterion,
        device=device
    )
    # 6. Save Model and Evaluate
    model_path = os.path.join(out_dir, "best_model.pth")
    model.save_model(model_path)
    
    # Save training logs
    logs_path = os.path.join(out_dir, "training_logs.json")
    with open(logs_path, "w") as f:
        json.dump({
            "train_losses": train_losses,
            "val_losses": val_losses,
            "lr_history": lr_history
        }, f, indent=4)
        
    print(f"Training finished successfully. Saved model and logs to {out_dir}")
if __name__ == "__main__":
    main()

