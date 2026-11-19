"""Vendored subset of curion-storage: only the lance row/task modules.

The upstream ``curion.storage.__init__`` eagerly imports the whole storage
contract, which pulls in ``curion.core``. Only ``lance_reader`` and
``lance_task`` are needed for training, and both depend on nothing but
pylance + pyarrow (see the module docstrings upstream), so this namespace
deliberately imports nothing.
"""
