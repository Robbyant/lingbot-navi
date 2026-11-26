"""Synchronous model process; heavy model dependencies stay outside Habitat."""
import json
import os
from pathlib import Path
import select
import subprocess


class ModelClient:
    def __init__(self, python, output, settings, timeout=900):
        self.timeout = timeout
        self.log = open(Path(output) / 'model.log', 'w')
        root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['OMP_NUM_THREADS'] = '4'
        env['MKL_NUM_THREADS'] = '4'
        env['OPENBLAS_NUM_THREADS'] = '1'
        self.process = subprocess.Popen(
            [str(python), '-u', '-m', 'close_loop_test.common.worker'], cwd=root, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1)
        self.sequence = 0
        try:
            self.identity = self.request('initialize', **settings)
        except BaseException:
            self.close()
            raise

    def request(self, operation, **fields):
        self.sequence += 1
        payload = dict(id=self.sequence, operation=operation, **fields)
        self.process.stdin.write(json.dumps(payload) + '\n')
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], self.timeout)
        if not ready:
            raise TimeoutError('Model request timed out; see model.log')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Model process exited; see model.log')
        response = json.loads(line)
        if response.get('id') != self.sequence:
            raise RuntimeError('Model protocol response ID mismatch')
        if not response.get('ok'):
            raise RuntimeError(response.get('error', 'Model request failed'))
        return response['result']

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()
