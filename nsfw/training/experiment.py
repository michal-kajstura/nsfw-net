from pathlib import Path
from typing import Dict, Any, List, Optional

import mlflow
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from nsfw.training.training import NsfwSystem


class Experiment:
    def __init__(self, checkpoint_path: Path, config: Dict[str, Any], transforms,
                 use_gpus: Optional[List[int]] = None):
        self._nsfw_system = NsfwSystem(config, transforms)

        checkpoint_callback = ModelCheckpoint(str(checkpoint_path),
                                              save_top_k=True,
                                              verbose=True,
                                              monitor='valid_loss',
                                              mode='min')
        self._trainer = Trainer(checkpoint_callback=checkpoint_callback,
                                gpus=use_gpus)

    def run(self):
        with mlflow.start_run():
            self._trainer.fit(self._nsfw_system)


