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
    name: str = "improved_cnn"
    num_classes: int = 4
    pretrained: bool = False


@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    ssl_enabled: bool = True


@dataclass
class SSLConfig:
    """Cấu hình cho Semi-Supervised Learning (Pseudo-Labeling + Curriculum).

    labeled_per_class  : Số ảnh có nhãn mỗi class (mặc định 250 → 1000 tổng).
    warmup_epochs      : Số epoch huấn luyện supervised trước khi bắt đầu SSL.
    ssl_epochs         : Số epoch SSL (pseudo-labeling).
    pseudo_threshold_start : Ngưỡng confidence ban đầu (curriculum bắt đầu thấp).
    pseudo_threshold_end   : Ngưỡng confidence cuối (curriculum tăng dần).
    pseudo_loss_weight     : λ — trọng số của loss từ pseudo-labeled data.
    min_pseudo_per_class   : Số pseudo-label tối thiểu mỗi class để bắt đầu SSL epoch.
    """
    labeled_per_class: int = 250
    warmup_epochs: int = 15
    ssl_epochs: int = 20
    pseudo_threshold_start: float = 0.85
    pseudo_threshold_end: float = 0.95
    pseudo_loss_weight: float = 0.5
    min_pseudo_per_class: int = 10


@dataclass
class ModeConfig:
    enabled: bool
    subset_ratio: float
    image_size: int
    batch_size: int
    epochs: int
    early_stopping_patience: int = 7


@dataclass
class Config:
    experiment_name: str
    project_name: str
    seed: int
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    ssl: SSLConfig
    development: ModeConfig
    research: ModeConfig

    @property
    def active_mode(self) -> ModeConfig:
        if self.development.enabled:
            return self.development
        return self.research

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg_dict = yaml.safe_load(f)

        # SSLConfig — dùng default nếu không có trong yaml
        ssl_cfg = SSLConfig(**cfg_dict.get('ssl', {})) if cfg_dict.get('ssl') else SSLConfig()

        return cls(
            experiment_name=cfg_dict.get('experiment_name', 'ssl_experiment'),
            project_name=cfg_dict.get('project_name', 'brain_tumor_detection'),
            seed=cfg_dict.get('seed', 42),
            data=DataConfig(**cfg_dict.get('data', {})),
            model=ModelConfig(**cfg_dict.get('model', {})),
            training=TrainingConfig(**cfg_dict.get('training', {})),
            ssl=ssl_cfg,
            development=ModeConfig(**cfg_dict.get('development', {})),
            research=ModeConfig(**cfg_dict.get('research', {})),
        )
