from typing import Dict, Optional, List, Tuple
import os, shutil

import wandb

from golem.utils import join_paths


class Logger:
    def __init__(
            self,
            metrics: List[Tuple[str, str]],
            log_mode: str = "wandb",  # 'local' or 'wandb'
            config: Optional[Dict] = None,
            wandb_kwargs: Optional[Dict] = None
    ):
        self.metrics = metrics
        self.log_mode = log_mode
        self.config = config
        self.wandb_kwargs = wandb_kwargs

        if log_mode == "wandb":
            wandb_kwargs["config"] = self.config
            if "mode" not in wandb_kwargs:
                wandb_kwargs["mode"] = "disabled"

            self.wandb_run: Optional[wandb.run] = wandb.init(reinit=True, **wandb_kwargs)
            self.wandb_dir = self.wandb_run.dir
            self.wandb_enabled = wandb_kwargs["mode"] != "disabled"

            for name, step_metric in metrics:
                self.wandb_run.define_metric(name, step_metric=step_metric)
        elif log_mode == "local":
            self.wandb_run: Optional[wandb.run] = None
            self.wandb_enabled = False
            raise NotImplementedError("log_mode = 'local' is not currently supported")
        else:
            raise ValueError(f"log_mode == {log_mode} is invalid")

    def add_metrics(self, metrics):
        if self.log_mode == "wandb":
            for name, step_metric in metrics:
                self.wandb_run.define_metric(name, step_metric=step_metric)

        self.metrics.extend(metrics)

    def log(self, d: Dict):
        if self.log_mode == "wandb":
            if self.wandb_enabled:
                self._log_wandb(d)
        else:
            raise NotImplementedError

    def log_file(self, file_path):
        if self.log_mode == "wandb":
            if self.wandb_enabled:
                self._log_file_wandb(file_path)
        else:
            raise NotImplementedError

    def log_artifact(
            self,
            file_path: str,
            name: Optional[str] = None,
            artifact_type: Optional[str] = None,
            aliases: Optional[List[str]] = None
    ) -> Optional[wandb.Artifact]:
        if self.log_mode == "wandb":
            if self.wandb_enabled:
                return self._log_artifact_wandb(file_path, name, artifact_type, aliases)
        else:
            raise NotImplementedError

    def _log_wandb(self, d: Dict):
        self.wandb_run.log(d)

    def _log_file_wandb(
            self,
            file_path: str
    ):
        file_name = os.path.basename(file_path)
        wandb_path = join_paths([self.wandb_dir, file_name])
        shutil.copy(file_path, wandb_path)
        self.wandb_run.save(wandb_path, base_path=self.wandb_dir)

    def _log_artifact_wandb(
            self,
            file_path: str,
            name: Optional[str] = None,
            artifact_type: Optional[str] = None,
            aliases: Optional[List[str]] = None
    ) -> wandb.Artifact:
        file_name = os.path.basename(file_path)
        wandb_path = join_paths([self.wandb_dir, file_name])
        shutil.copy(file_path, wandb_path)
        return self.wandb_run.log_artifact(wandb_path, name=name, type=artifact_type, aliases=aliases)

    def finish(self):
        if self.log_mode == "wandb":
            self.wandb_run.finish()
