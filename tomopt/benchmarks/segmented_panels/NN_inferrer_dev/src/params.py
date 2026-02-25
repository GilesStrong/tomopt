import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict

@dataclass
class ModelParams:
    """Parameters for the model architecture."""
    voxel_shape: tuple = (10, 10, 4)  # Default shape
    radius: float = 0.1
    device: str = "cuda"
    
@dataclass
class DataParams:
    """Parameters for data loading and splitting."""
    n_positions: int = 49
    n_materials: int = 5
    n_repetitions: int = 20
    val_combos: int = 200
    test_combos: int = 200
    random_seed: int = 32
    batch_size: int = 20
    
@dataclass
class TrainParams:
    """Parameters for the training loop."""
    n_epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-3
    patience: int = 3
    min_lr: float = 1e-6
    factor: float = 0.3
    loss_type: str = "huber"
    use_wandb: bool = False
    wandb_config: Optional[Dict] = None
    
@dataclass
class ExperimentParams:
    """Master configuration containing all parameters."""
    model: ModelParams = field(default_factory=ModelParams)
    data: DataParams = field(default_factory=DataParams)
    train: TrainParams = field(default_factory=TrainParams)
    
    @classmethod
    def from_json(cls, path: str):
        """Load parameters from a JSON file."""
        with open(path, 'r') as f:
            config = json.load(f)
            
        model_params = ModelParams(**config.get('model', {}))
        data_params = DataParams(**config.get('data', {}))
        train_params = TrainParams(**config.get('train', {}))
        
        return cls(model=model_params, data=data_params, train=train_params)
        
    def to_json(self, path: str):
        """Save parameters to a JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)
