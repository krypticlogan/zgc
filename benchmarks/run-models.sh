#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository_root=$(dirname -- "$script_dir")

cd "$repository_root"

for batch in 1 8 32 128; do
    for model in small medium large; do
        printf '\n%s\n' "Running model=${model} batch=${batch}"
        zig build benchmark \
            -Dop=model \
            -Dmodel="$model" \
            -Dbatch="$batch" \
            -Doptimize=ReleaseFast \
            "$@"
    done
done
