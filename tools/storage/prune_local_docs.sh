#!/bin/bash
# Local file management only. No Python, model, parser or analysis runs locally.
set -euo pipefail
workspace_root=${1:?workspace root required}
manifest=${2:?reviewed path list required}
receipt=${3:?new deletion receipt required}
test "$workspace_root" = '/Users/hao/Documents/ChatGPT/挑战赛'
cd "$workspace_root"
test -s ncccu_code/records/storage/EXPERIMENT-CONCLUSIONS.md
test -s ncccu_code/records/storage/EXPERIMENT-CONCLUSIONS.json
test ! -e "$receipt"
: > "$receipt"
while IFS= read -r retired_path; do
  case "$retired_path" in
    *'/../'*|*'/resume-9010-20261028/'*) exit 3 ;;
    work/auroc998-20261025/*.md|work/auroc999-20261027/*.md) ;;
    *) exit 4 ;;
  esac
  test ! -L "$retired_path"
  if [ -f "$retired_path" ]; then
    rm -- "$retired_path"
    printf '%s\n' "$retired_path" >> "$receipt"
  fi
done < "$manifest"
wc -l "$receipt"
