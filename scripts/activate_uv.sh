#!/usr/bin/env bash
# Source this file from the repository root to use the UV environment required by VLearn:
#   source scripts/activate_uv.sh

_transformerrl_root="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")/.." && pwd)"
source "$_transformerrl_root/.venv/bin/activate"

# VLearn dynamically loads both its bundled TurboActivate library and CUDA 13
# by soname. UV installs both inside the virtualenv, so expose both directories
# without relying on an old Conda environment or system-wide CUDA installation.
_transformerrl_site_packages="$(python -c \
  'import sysconfig; print(sysconfig.get_path("purelib"))')"
_transformerrl_cuda_lib="$_transformerrl_site_packages/nvidia/cu13/lib"
_transformerrl_vlearn_lib="$_transformerrl_site_packages/vlearn/lib"
export LD_LIBRARY_PATH="$_transformerrl_cuda_lib:$_transformerrl_vlearn_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# The native simulator resolves assets and its persistent cache relative to this directory.
# vlearn-main is the compatibility symlink created during setup, pointing at VLearn 0.3.12.
_transformerrl_vlearn_root="$(cd "$_transformerrl_root/../vlearn-main" && pwd -P)"
export VL_WORKING_DIRECTORY="$_transformerrl_vlearn_root"

unset _transformerrl_root _transformerrl_site_packages _transformerrl_cuda_lib
unset _transformerrl_vlearn_lib _transformerrl_vlearn_root
