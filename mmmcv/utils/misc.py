# Copyright (c) OpenMMLab. All rights reserved.
import collections.abc
import functools
import itertools
import subprocess
import warnings
import mmmcv
import logging
import os
from collections import abc
from importlib import import_module
from inspect import getfullargspec
from itertools import repeat
from mmmcv.utils import print_log


def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(
            f'custom_imports must be a list but got type {type(imports)}')
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(
                f'{imp} is of type {type(imp)} and cannot be imported.')
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f'{imp} failed to import and is ignored.',
                              UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


def update_data_root(cfg, logger=None):
    """Update data root according to env MMDET_DATASETS.

    If set env MMDET_DATASETS, update cfg.data_root according to
    MMDET_DATASETS. Otherwise, using cfg.data_root as default.

    Args:
        cfg (mmcv.Config): The model config need to modify
        logger (logging.Logger | str | None): the way to print msg
    """
    assert isinstance(cfg, mmmcv.Config), \
        f'cfg got wrong type: {type(cfg)}, expected mmcv.Config'

    if 'MMDET_DATASETS' in os.environ:
        dst_root = os.environ['MMDET_DATASETS']
        print_log(f'MMDET_DATASETS has been set to be {dst_root}.'
                  f'Using {dst_root} as data root.')
    else:
        return

    assert isinstance(cfg, mmmcv.Config), \
        f'cfg got wrong type: {type(cfg)}, expected mmcv.Config'

    def update(cfg, src_str, dst_str):
        for k, v in cfg.items():
            if isinstance(v, mmmcv.ConfigDict):
                update(cfg[k], src_str, dst_str)
            if isinstance(v, str) and src_str in v:
                cfg[k] = v.replace(src_str, dst_str)

    update(cfg.data, cfg.data_root, dst_root)
    cfg.data_root = dst_root
