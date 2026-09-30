import time
import re
import warnings
from typing import List

import psutil
import subprocess
import logging
import threading

import odyssey.utils as U


class SubprocessMonitor:
    def __init__(
        self,
        commands: List[str],
        name: str,
        ready_match: str = r".*",
        log_path: str = "logs",
        callback_match: str = r"^(?!x)x$",  # regex that will never match
        callback: callable = None,
        finished_callback: callable = None,
    ):
        self.commands = commands
        start_time = time.strftime("%Y%m%d_%H%M%S")
        self.name = name
        self.logger = logging.getLogger(name)
        self.handler = logging.FileHandler(U.f_join(log_path, f"{start_time}.log"))
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.handler.setFormatter(formatter)
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)
        self.process = None
        self.ready_match = ready_match
        self.ready_event = None
        self.ready_line = None
        self.callback_match = callback_match
        self.callback = callback
        self.finished_callback = finished_callback
        self.thread = None

    def _start(self):
        self.logger.info(f"Starting subprocess with commands: {self.commands}")
        if self.process is not None:
            self._stop_process(self.process)
        process = psutil.Popen(
            self.commands,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        self.process = process
        print(f"Subprocess {self.name} started with PID {process.pid}.")
        try:
            for line in iter(process.stdout.readline, ""):
                self.logger.info(line.strip())
                if re.search(self.ready_match, line):
                    self.ready_line = line
                    self.logger.info("Subprocess is ready.")
                    self.ready_event.set()
                if re.search(self.callback_match, line):
                    self.callback()
        finally:
            if process.stdout:
                process.stdout.close()
            if not self.ready_event.is_set():
                self.ready_event.set()
                warnings.warn(f"Subprocess {self.name} failed to start.")
            if self.finished_callback:
                self.finished_callback()

    def run(self):
        self.ready_event = threading.Event()
        self.ready_line = None
        self.thread = threading.Thread(target=self._start, name=f"{self.name}-stdout", daemon=True)
        self.thread.start()
        # block until read_event is set
        print('wait ready_event to set')
        # is it possible that the ready_event is never set?
        # self.ready_event.wait(timeout=60)
        self.ready_event.wait()

    def stop(self):
        self.logger.info("Stopping subprocess.")
        process = self.process
        if process is not None:
            self._stop_process(process)
            if self.process is process:
                self.process = None
        thread = self.thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=10)
        if thread is not None and thread is self.thread and not thread.is_alive():
            self.thread = None

    def close(self):
        self.stop()
        if self.handler:
            self.logger.removeHandler(self.handler)
            self.handler.close()
            self.handler = None

    @staticmethod
    def _stop_process(process):
        if process.is_running():
            process.terminate()
            try:
                process.wait(timeout=10)
            except psutil.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if process.stdout:
            process.stdout.close()

    # def __del__(self):
    #     if self.process.is_running():
    #         self.stop()

    @property
    def is_running(self):
        if self.process is None:
            return False
        # TODO: 
        # if self.process.is_running() and self.ready_line is None:
        #     self.stop()
        #     raise RuntimeError('Subprocess is running but ready_line is None. It may mean that the process has not started yet.')
        return self.process.is_running()
