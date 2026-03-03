import os

import nqxpack
from advanced_drivers._src.callbacks.base import AbstractCallback
from netket.utils import struct


class SaveStatesCallback(AbstractCallback, mutable=True):
    r"""
    Periodically saves `driver.state_p` and `driver.state_q` to disk with
    nqxpack.  Files are written to `out_dir/state_p_step{step}.mpack` and
    `out_dir/state_q_step{step}.mpack`.

    Parameters
    ----------
    out_dir      : directory in which checkpoints are written
    save_every   : checkpoint interval (default: every 100 steps)
    keep_last_n  : if > 0, only keep the `keep_last_n` most recent
                   checkpoints (older files are deleted).  0 = keep all.
    """

    _out_dir: str = struct.field(pytree_node=False)
    _save_every: int = struct.field(pytree_node=False, default=100)
    _keep_last_n: int = struct.field(pytree_node=False, default=0)
    _saved_steps: list = struct.field(pytree_node=False, default=None)

    def __init__(self, out_dir: str, save_every: int = 100, keep_last_n: int = 0):
        super().__init__()
        os.makedirs(out_dir, exist_ok=True)
        self._out_dir = out_dir
        self._save_every = save_every
        self._keep_last_n = keep_last_n
        self._saved_steps: list[int] = []

    def on_compute_update_end(self, step, log_data, driver):
        if step % self._save_every != 0:
            return

        path_p = os.path.join(self._out_dir, f"state_p_step{step}.mpack")
        path_q = os.path.join(self._out_dir, f"state_q_step{step}.mpack")

        nqxpack.save(driver.state_p, path_p)
        nqxpack.save(driver.state_q, path_q)
        self._saved_steps.append(step)

        # Prune old checkpoints
        if self._keep_last_n > 0 and len(self._saved_steps) > self._keep_last_n:
            old_step = self._saved_steps.pop(0)
            for suffix in ("state_p", "state_q"):
                old_path = os.path.join(self._out_dir, f"{suffix}_step{old_step}.mpack")
                if os.path.exists(old_path):
                    os.remove(old_path)