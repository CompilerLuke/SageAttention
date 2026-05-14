#!/usr/bin/env bash
# Source this before building when CUDA headers come from pip NVIDIA packages.
#
# Usage:
#   source scripts/export_cuda_headers.sh
#   cd sageattention3_blackwell
#   python setup.py build_ext --inplace --fast-build-specialized

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This script must be sourced so CPATH is exported in your current shell:" >&2
  echo "  source ${BASH_SOURCE[0]}" >&2
  exit 2
fi

_sage_cuda_include="$(
python - <<'PY'
from pathlib import Path
import site
import sys

roots = []
for getter in (site.getsitepackages,):
    try:
        roots.extend(getter())
    except Exception:
        pass
try:
    roots.append(site.getusersitepackages())
except Exception:
    pass
roots.extend(sys.path)

seen = set()
for root in roots:
    if not root:
        continue
    base = Path(root)
    if base in seen:
        continue
    seen.add(base)
    for include_dir in sorted(base.glob("nvidia/cu*/include"), reverse=True):
        if (include_dir / "cuda_runtime.h").exists():
            print(include_dir)
            raise SystemExit(0)

raise SystemExit(1)
PY
)"

if [[ -z "${_sage_cuda_include}" ]]; then
  echo "Could not find pip NVIDIA CUDA headers." >&2
  echo "Install them with: python -m pip install 'nvidia-cuda-runtime>=13,<14'" >&2
  return 1
fi

case ":${CPATH:-}:" in
  *:"${_sage_cuda_include}":*) ;;
  *) export CPATH="${_sage_cuda_include}${CPATH:+:${CPATH}}" ;;
esac

export CUDA_HEADER_INCLUDE_DIR="${_sage_cuda_include}"
echo "Exported CUDA headers: ${CUDA_HEADER_INCLUDE_DIR}"
echo "CPATH=${CPATH}"

unset _sage_cuda_include
