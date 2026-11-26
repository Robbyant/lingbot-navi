"""Synchronous IPC client used only by the bounded export consumer thread."""
import os
import atexit
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

from .protocol import send, receive


class PipeClient:
    def __init__(self, python, *, env):
        self.python, self.env = str(python), env
        self.process = None
        self.script = Path(__file__).with_name("worker.py")
        atexit.register(self.close)

    def request(self, message):
        if self.process is None or self.process.poll() is not None:
            if message['op'] != 'begin':
                raise RuntimeError('online memory worker died; segment cannot be resumed')
            self.process = subprocess.Popen([self.python, str(self.script)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, env=self.env)
        try:
            send(self.process.stdin, message)
            result = receive(self.process.stdout)
        except (EOFError, BrokenPipeError, OSError):
            self.close()
            raise RuntimeError('online memory worker connection lost')
        if not result.get('ok'):
            raise RuntimeError(result.get('error', 'online memory worker failed'))
        return result

    def close(self):
        process, self.process = self.process, None
        if process is None:
            return
        if process.stdin:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        if process.stdout:
            process.stdout.close()


def map1_builder_rgb(rgb):
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    width, height = image.size
    height = round(height * 518 / width / 14) * 14
    image = image.resize((518, height), Image.Resampling.BICUBIC)
    if height > 518:
        top = (height - 518) // 2
        image = image.crop((0, top, 518, top + 518))
    return np.array(image, dtype=np.uint8)


class Map1MemoryClient(PipeClient):
    def __init__(self, python, checkpoint, device):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        if device.startswith("cuda:"):
            index = int(device.split(":", 1)[1])
            visible = env.get("CUDA_VISIBLE_DEVICES")
            env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[index] if visible is not None else str(index)
        super().__init__(python, env=env)
        self.checkpoint = str(checkpoint)

    def begin_stream(self):
        return self.request(dict(op="begin", streaming=True, checkpoint=self.checkpoint))
