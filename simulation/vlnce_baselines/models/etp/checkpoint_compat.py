"""Normalize RPGNav checkpoint keys before model loading."""
from collections import OrderedDict
from collections.abc import Mapping


def _rename(key):
    parts = key.split('.')
    for index, part in enumerate(parts):
        for old, new in (('st_planner', 'rp_planner'),
                         ('ST_PLANNER', 'RP_PLANNER'),
                         ('st_progress', 'rp_progress')):
            if part == old or part.startswith(old + '_'):
                parts[index] = new + part[len(old):]
                break
    return '.'.join(parts)


def rename_rp_state_dict(state):
    """Rename state keys, including DDP prefixes and PyTorch module metadata."""
    renamed = OrderedDict()
    for key, value in state.items():
        target = _rename(key)
        if target in renamed:
            raise ValueError('Ambiguous legacy/current checkpoint keys: ' + target)
        renamed[target] = value
    if hasattr(state, '_metadata'):
        renamed._metadata = {_rename(k): v for k, v in state._metadata.items()}
    return renamed


def rename_rp_config(config):
    """Normalize archived deployment config mappings without importing Habitat."""
    if not isinstance(config, Mapping):
        return config
    renamed = {}
    for key, value in config.items():
        target = _rename(key)
        if target in renamed:
            raise ValueError('Ambiguous legacy/current config keys: ' + target)
        renamed[target] = rename_rp_config(value)
    return renamed
