import yaml
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class DataConfig:
    dataset1_train_path: str
    dataset1_test_path: str
    dataset2_path: str
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4

@dataclass
class ModelConfig:
    name: str = "resnet18"
    num_classes: int = 4
    pretrained: bool = True

@dataclass
class TrainingConfig:
    epochs: int = 30
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    ssl_enabled: bool = False

@dataclass
class Config:
    experiment_name: str
    project_name: str
    seed: int
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        with open(yaml_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
            
        return cls(
            experiment_name=cfg_dict.get('experiment_name', 'default_experiment'),
            project_name=cfg_dict.get('project_name', 'brain_tumor_detection'),
            seed=cfg_dict.get('seed', 42),
            data=DataConfig(**cfg_dict.get('data', {})),
            model=ModelConfig(**cfg_dict.get('model', {})),
            training=TrainingConfig(**cfg_dict.get('training', {}))
        )
