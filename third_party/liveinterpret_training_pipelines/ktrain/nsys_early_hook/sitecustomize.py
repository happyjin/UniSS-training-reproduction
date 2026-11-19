# Copyright (c) junyaoyang. All rights reserved.
"""Auto-imported by CPython's ``site`` module at interpreter startup —
i.e. BEFORE ``swift.cli.sft`` or any other project code is imported. This
is the *only* hook point early enough to patch Ray's actor-creation
machinery before ``swift.ray_utils.base.RayHelper``'s ``@worker(...)``
class decorator (applied to ``SwiftSft`` when ``swift.pipelines.train.sft``
is first imported) calls ``ray.remote(cls)`` — see
``custom/ray_nsight_patch.py``'s module docstring for the full root-cause
writeup of why patching from ``--external_plugins`` (``custom/plugin.py``)
is fundamentally too late (``RayHelper._create_workers()`` already runs
*before* ``--external_plugins`` gets parsed, per ``swift.ray_utils.base.
RayHelper.worker()``'s ``new_init`` call order).

This file does nothing unless ``NSYS_PROFILE_STEP`` and ``USE_RAY`` are
both set (same switches as ``custom/ray_nsight_patch.py`` /
``custom/nsys_step_profile.py``) — safe no-op for every other python
invocation that happens to inherit this directory on its ``PYTHONPATH``
(e.g. ``ktrain`` CLI's own subprocess calls, ``sync_code.sh``'s python
one-liners, etc).

Picked up automatically because this directory is prepended to
``PYTHONPATH`` by ``ktrain/launch_train.sh`` / ``ktrain/
launch_train_lance.sh``. Do not rename this file — ``sitecustomize`` is a
magic module name recognized by CPython's ``site`` module; this only works
because the filename matches exactly.
"""
import os

_step = os.environ.get('NSYS_PROFILE_STEP', '0')
_use_ray = os.environ.get('USE_RAY', 'false').strip().lower() in ('1', 'true')

if _step not in ('', '0') and _use_ray:
    try:
        from custom.ray_nsight_patch import install_ray_nsight_patch
        install_ray_nsight_patch()
    except Exception as e:  # pragma: no cover - never break interpreter startup
        print(f'[sitecustomize][ERROR] failed to install ray nsight patch '
              f'at interpreter startup: {e!r}', flush=True)
