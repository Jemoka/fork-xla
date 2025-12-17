# common standard library utilities
import os
import sys
import time
import json
import math
import random
from random import Random
from collections import defaultdict

from pathlib import Path
from argparse import Namespace

# machine learning and data utilities
import numpy as np
import pandas as pd

# MLOps
import wandb

# logging
import logging
from loguru import logger

# to contextualize plotting
from contextlib import contextmanager

# plot function
from plots import plot as do_plot
from collections.abc import Callable

from io import BytesIO, SEEK_SET, SEEK_END


# async dispatch
from threading import Thread
from collections.abc import Callable

import requests, tarfile, io, os, zipfile, gzip
from typing import Optional
import tomllib

from io import BytesIO, SEEK_SET, SEEK_END
from tqdm import tqdm

from flywheel import Sampling, Strategy




R = Random(7)


class PlotLoggingHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        self.args = kwargs.pop("args")
        self.logger = kwargs.pop("logger")
        super().__init__(*args, **kwargs)

        # because we will plot() whenever, but only plot
        # specificlally when we actually want to plot
        self.__write_plot = False
        # things yet to be plotted --- plot_name : {key : value}
        # we plot only when the plot() function is called
        self.__cached_plots = defaultdict(lambda: defaultdict(dict))
        # we truly save every plot now
        # {plot: [(plot, idx)]}
        self.__saved_plots = defaultdict(list)

    @property
    def plots(self):
        return dict(self.__saved_plots)

    def arm(self):
        self.__write_plot = True

    def plot(self, idx=None, override_plot_funcs={}, debug=True):
        # actually emit the plots
        logs = {}
        for k, v in self.__cached_plots.items():
            v = {
                i: {b: c() if isinstance(c, Callable) else c for b, c in a.items()}
                if isinstance(a, dict)
                else a
                for i, a in v.items()
            }
            if override_plot_funcs.get(k):
                plotted = override_plot_funcs.get(k)(v)
            else:
                plotted = do_plot(k, v)
            if not plotted:
                continue
            if isinstance(plotted, dict):
                # we want multiple plots from each plot function
                for a, b in plotted.items():
                    (save, log) = b
                    logs[a] = log
                    if debug:
                        self.__saved_plots[a].append((save, idx))
            else:
                (save, log) = plotted
                logs[k] = log
                if debug:
                    self.__saved_plots[k].append((save, idx))
        self.logger(logs, step=idx)

        # flush the cache
        self.__cached_plots = defaultdict(lambda: defaultdict(dict))
        self.__write_plot = False

    def emit(self, record: logging.LogRecord, debug=True) -> None:
        name = record.getMessage()
        kwargs = record.extra["payload"]
        key = record.extra["key"]

        if self.__write_plot:
            if key == None:
                if self.__cached_plots.get(name):
                    logger.warning(
                        "Plot already exists before flush! Make sure that "
                        "if you are plotting multiple things with the same "
                        "name within a single plot context, that the have "
                        "distinct keys; overwriting the past one!!!; duplicated "
                        "name: {}",
                        name,
                    )
                self.__cached_plots[name] = kwargs
            else:
                self.__cached_plots[name][key] = kwargs


def plot_logger(*args, **kwargs):
    handler = PlotLoggingHandler(*args, **kwargs)
    logger.add(
        handler,
        filter=lambda x: x["extra"].get("task", "") == "plot",
        format="{message}",
    )

    @contextmanager
    def emit(idx=None, override_plot_funcs={}, debug=True):
        handler.arm()
        try:
            yield
        finally:
            # this will actually emit the plots
            handler.plot(idx, override_plot_funcs, debug=debug)

    def get_plots():
        return handler.plots

    return emit, get_plots


def plot(name, key=None, **kwargs):
    logger.bind(task="plot", payload=kwargs, key=key).info(name)

class UrlStream:
    def __init__(self, url, headers={}):
        self._url = url
        self._head = headers
        rq = requests.head(url, headers={"Accept-Encoding": "identity", **self._head})
        if rq.status_code == 302:
            self._url = rq.headers['location']
            rq = requests.head(self._url, headers={"Accept-Encoding": "identity", **self._head})

        headers = {k.lower(): v for k, v in rq.headers.items()}
        self._seek_supported = headers.get('accept-ranges') == 'bytes' and 'content-length' in headers
        if self._seek_supported:
            self._size = int(headers['content-length'])
        self._curr_pos = 0
        self._buf_start_pos = 0
        self._iter = None
        self._buffer = None
        self._buf_size = 0
        self._loaded_all = False

    def _load_all(self):
        if self._loaded_all:
            return
        self._make_request()
        old_buf_pos = self._buffer.tell()
        self._buffer.seek(0, SEEK_END)
        for chunk in self._iter:
            self._buffer.write(chunk)
        self._buf_size = self._buffer.tell()
        self._buffer.seek(old_buf_pos, SEEK_SET)
        self._loaded_all = True

    def seekable(self):
        return self._seek_supported

    def seek(self, position, whence=SEEK_SET):
        if whence == SEEK_END:
            assert position <= 0
            if self._seek_supported:
                self.seek(self._size + position)
            else:
                self._load_all()
                self._buffer.seek(position, SEEK_END)
                self._curr_pos = self._buffer.tell()
        elif whence == SEEK_SET:
            if self._curr_pos != position:
                self._curr_pos = position
                if self._seek_supported:
                    self._iter = None
                    self._buffer = None
                else:
                    self._load_until(position)
                    self._buffer.seek(position)
                    self._curr_pos = position
        else:
            assert "Invalid whence %s" % whence

        return self.tell()

    def tell(self):
        return self._curr_pos

    def _load_until(self, goal_position):
        retries = 0

        while True:
            try:
                self._make_request()
                old_buf_pos = self._buffer.tell()
                self._buffer.seek(0, SEEK_END)

                goal_position = goal_position - self._buf_start_pos
                while self._buf_size < goal_position:
                    try:
                        d = next(self._iter)
                        self._buffer.write(d)
                        self._buf_size += len(d)
                    except StopIteration:
                        break
                self._buffer.seek(old_buf_pos, SEEK_SET)
            except requests.exceptions.ChunkedEncodingError as e:
                # This happens if the server is overloaded, in the next() call from above. Restore buffer, reset the
                # connection and try again. Unfortunately we can't hook into requests to partial data downloaded,
                # and also we can't use the data in it's internal buffer.

                if retries >= 5:
                    print(f"Multiple consecutive failures trying to download a chink from {self._url}. Giving up.")
                    raise

                retries += 1
                self._buffer.seek(old_buf_pos, SEEK_SET)
                self._iter = None
            else:
                break

    def _new_buffer(self):
        remaining = self._buffer.read() if self._buffer is not None else None
        self._buffer = BytesIO()
        if remaining is not None:
            self._buffer.write(remaining)
        self._buf_start_pos = self._curr_pos
        self._buf_size = 0 if remaining is None else len(remaining)
        self._buffer.seek(0, SEEK_SET)
        self._loaded_all = False

    def _make_request(self):
        if self._iter is None:
            h = {
                "User-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.80 Safari/537.36",
                **self._head
            }
            if self._seek_supported:
                h["Range"] = "bytes=%d-%d" % (self._curr_pos + (self._buf_size if self._buffer is not None else 0), self._size - 1)

            r = requests.get(self._url, headers=h, stream=True)

            self._iter = r.iter_content(1024 * 1024)
            self._new_buffer()
        elif self._seek_supported and self._buf_size > 128 * 1024 * 1024:
            self._new_buffer()

    def size(self):
        if self._seek_supported:
            return self._size
        else:
            self._load_all()
            return self._buf_size

    def read(self, size=None):
        if size is None:
            size = self.size()

        self._load_until(self._curr_pos + size)
        if self._seek_supported:
            self._curr_pos = min(self._curr_pos + size, self._size)

        read_data = self._buffer.read(size)
        if not self._seek_supported:
            self._curr_pos += len(read_data)
        return read_data

    def iter_content(self, block_size):
        while True:
            d = self.read(block_size)
            if not len(d):
                break
            yield d


def download(url: str, dest: Optional[str] = None, extract: bool=True, ignore_if_exists: bool = False,
             compression: Optional[str] = None, headers={}):
    """
    Download a file from the internet.

    Args:
        url: the url to download
        dest: destination file if extract=False, or destionation dir if extract=True. If None, it will be the last part of URL.
        extract: extract a tar.gz or zip file?
        ignore_if_exists: don't do anything if file exists

    Returns:
        the destination filename.
    """

    base_url = url.split("?")[0]
    url_end = [f for f in base_url.split("/") if f][-1]

    if dest is None:
        dest = url_end

    stream = UrlStream(url, headers=headers)
    extension = base_url.split(".")[-1].lower()

    if extract and extension in ['gz', 'bz2', 'zip', 'tgz', 'tar']:
        if os.path.exists(dest) and ignore_if_exists:
            return dest

        os.makedirs(dest, exist_ok=True)

        if extension == "gz" and not base_url.endswith(".tar.gz"):
            decompressed_file = gzip.GzipFile(fileobj=stream)
            with open(os.path.join(dest, url.split("/")[-1][:-3]), 'wb') as f:
                while True:
                    d = decompressed_file.read(1024 * 1024)
                    if not d:
                        break
                    f.write(d)
        else:
            if extension in ['gz', 'bz2', "tgz", "tar"]:
                decompressed_file = tarfile.open(fileobj=stream, mode='r|' +
                                                                      (compression or (
                                                                          "gz" if extension == "tgz" else extension)))
            elif extension == 'zip':
                decompressed_file = zipfile.ZipFile(stream, mode='r')
            else:
                assert False, "Invalid extension: %s" % extension

            decompressed_file.extractall(dest)
    else:
        if dest.endswith("/"):
            dest = os.path.join(dest, url_end)

        if os.path.exists(dest) and ignore_if_exists:
            return dest

        try:
            p = tqdm(total=stream.size())
            with open(dest, 'wb') as f:
                for d in stream.iter_content(1024 * 1024):
                    f.write(d)
                    p.update(len(d))

        except:
            os.remove(dest)
            raise
    return dest


zstd = None
def read_lines_from_zst(url: str, max_retries: int = 5, initial_backoff: float = 1.0):
    # from https://stackoverflow.com/questions/61067762/how-to-extract-zst-files-into-a-pandas-dataframe
    import itertools
    global zstd
    if zstd is None:
        import zstandard as zstd

    urls = UrlStream(url)
    DCTX = zstd.ZstdDecompressor(max_window_size=2**31)

    iterator = None
    retry_count = 0
    backoff = initial_backoff
    lines_yielded = 0  # Track how many lines we've successfully yielded

    while True:
        try:
            if iterator is None:
                zfh = zstd.open(urls, mode='rb', dctx=DCTX)
                iofh = io.TextIOWrapper(zfh)
                iterator = iter(iofh)
                # Skip lines we've already yielded (efficient with islice)
                if lines_yielded > 0:
                    for _ in itertools.islice(iterator, lines_yielded):
                        pass

            line = next(iterator)
            retry_count = 0  # Reset retry count on success
            backoff = initial_backoff  # Reset backoff on success
            lines_yielded += 1
            yield line

        except StopIteration:
            break
        except Exception as e:
            # Catch ZstdError and other decompression errors
            if 'ZstdError' in type(e).__name__ or isinstance(e, (IOError, OSError)):
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Failed to read from {url} after {max_retries} retries")
                    raise
                logger.warning(f"ZstdError encountered at line {lines_yielded}, retry {retry_count}/{max_retries} after {backoff}s")
                time.sleep(backoff)
                backoff *= 2  # Exponential backoff
                # Reset the iterator and we'll skip to where we were
                iterator = None
                urls = UrlStream(url)
            else:
                raise


def parse_dataset_spec(path_to_spec: str, args):
    """Parse a TOML dataset specification file and return Strategy.

    Expected TOML format:
    ```toml
    [[dataset]]
    type = "memmap"
    path = "/path/to/dataset"
    weight = 0.5

    [[dataset]]
    type = "memmap"
    path = "/path/to/another"
    weight = 0.5
    ```

    Returns:
        Strategy object
    """
    from flywheel import MemmapDataset

    # Registry mapping type strings to init functions
    # init_function takes (args, config_dict) and returns dataset instance
    DATASET_TYPES = {
        'memmap': lambda args, cfg: MemmapDataset(args, cfg['path'], has_val=cfg.get("has_val", False)),
        # Add new dataset types here:
        # 'huggingface': lambda args, cfg: HFDataset(args, cfg['name'], cfg.get('split', 'train')),
        # 'custom': lambda args, cfg: CustomDataset(args, **{k: v for k, v in cfg.items() if k not in ['type', 'weight']}),
    }

    with open(path_to_spec, 'rb') as f:
        config = tomllib.load(f)

    datasets = config.get('dataset', [])
    if not datasets:
        raise ValueError("No datasets found in spec")

    sampling_objs = []
    for ds_config in datasets:
        ds_type = ds_config.get('type')
        weight = ds_config.get('weight', 1.0)

        if ds_type not in DATASET_TYPES:
            raise ValueError(f"Unknown dataset type: {ds_type}. Available types: {list(DATASET_TYPES.keys())}")

        init_fn = DATASET_TYPES[ds_type]
        dataset = init_fn(args, ds_config)
        sampling_objs.append(Sampling(dataset, weight))

    return Strategy(args, sampling_objs)
