from typing import Optional

from mmcv.runner.dist_utils import master_only
from mmcv.runner import HOOKS
from mmcv.runner import LoggerHook

from dvclive import Live


@HOOKS.register_module()
class DVCLoggerHook(LoggerHook):
    def __init__(self,
                log_dir: Optional[str] = None,
                interval: int = 10,
                ignore_last: bool = True,
                reset_flag: bool = False,
                by_epoch: bool = True):
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.log_dir = log_dir
        

    @master_only
    def before_run(self, runner) -> None:
        super().before_run(runner)
        #if self.log_dir is None:
        #    self.log_dir = osp.join(runner.work_dir, 'dvc_logs')
        self.logger = Live(dir="eval", report="auto")

    @master_only
    def log(self, runner) -> None:
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, float):
                self.logger.log_metric(tag, val)

    @master_only
    def after_run(self, runner) -> None:
        self.logger.make_report()