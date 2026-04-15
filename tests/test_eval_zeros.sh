#!/bin/bash
set -euo pipefail

IMAGE=mininnverifier

# circles_network_5 has one input of shape [2]: 2 float64 values = 16 zero bytes
dd if=/dev/zero bs=16 count=1 of=zero_input.bin 2>/dev/null

docker run --rm \
  -v "$(pwd):/tests" \
  "$IMAGE" eval \
  /tests/resources/circles_network_5.mininn \
  /tests/zero_input.bin |
  od -A none -t f8

rm zero_input.bin
