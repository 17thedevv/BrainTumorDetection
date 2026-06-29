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
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    ssl_enabled: bool = False

@dataclass
class ModeConfig:
    enabled: bool
    subset_ratio: float
    image_size: int
    batch_size: int
    epochs: int

@dataclass
class Config:
    experiment_name: str
    project_name: str
    seed: int
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    development: ModeConfig
    research: ModeConfig
    
    @property
    def active_mode(self) -> ModeConfig:
        if self.development.enabled:
            return self.development
        return self.research

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
            training=TrainingConfig(**cfg_dict.get('training', {})),
            development=ModeConfig(**cfg_dict.get('development', {})),
            research=ModeConfig(**cfg_dict.get('research', {}))
        )
