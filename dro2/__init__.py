"""Compatibility layer for legacy ``dro_new`` imports."""

import sys

module = sys.modules[__name__]
# Expose this package under the historical name ``dro_new`` so that modules
# importing ``dro_new.ascdro`` continue to work after the directory rename.
sys.modules.setdefault("dro_new", module)

__all__ = []
