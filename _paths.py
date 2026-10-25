"""Put every code/ sub-package on sys.path.

Imported for its side effect by the scripts in common/, figures/, analysis/,
mapping/ and tools/ so that

  * a script can be run from any working directory, and
  * a module can be dropped into whichever folder matches its purpose without
    touching the import lines of anything that already exists.

Directories are listed explicitly rather than discovered, so a stray folder
(for instance a virtualenv) can never shadow a module of the same name.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

SUBDIRS = (
    "common",     # shared libraries (mapping, Bernoulli operators, panel method)
    "figures",    # manuscript figure generators
    "analysis",   # scripts backing text/table numbers
    "mapping",    # conformal-mapping / Theodorsen / Laurent pipeline
    "tools",      # panel splitting and reproduction checks
    "real_device",  # hardware experiments
    os.path.join("real_device", "sm"),           # QFT-propagator line
    os.path.join("real_device", "vqe"),          # variational encoder
    os.path.join("real_device", "minmod"),       # quantum Minmod line
    os.path.join("real_device", "calibration"),  # chip calibration / cluster search
)

for _name in (None,) + SUBDIRS:
    _p = ROOT if _name is None else os.path.join(ROOT, _name)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
