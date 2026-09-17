#!/usr/bin/env python3

from pathlib import Path
from string import Template


NUM_WARPS = 8
NUM_CORES = 2
ITERS = 4096
UNROLL = 16
RESULT_WORDS = NUM_WARPS * NUM_CORES
PAYLOAD = 0x13579BDF


def main() -> None:
    data_template = Template(
        """
__global PointerChaseNode pointerchase_nodes[] = {
    { nullptr, $payload },
};
"""
    )

    expected_template = Template(
        """
iters: $iters
unroll: $unroll
num_warps: $num_warps
num_cores: $num_cores
result_words: $result_words

expected_pointerchase_sink: pointerchase_nodes
"""
    )

    data_output = data_template.substitute(
        payload=f"0x{PAYLOAD:08x}",
    ).lstrip()

    expected_output = expected_template.substitute(
        iters=ITERS,
        unroll=UNROLL,
        num_warps=NUM_WARPS,
        num_cores=NUM_CORES,
        result_words=RESULT_WORDS,
    ).lstrip()

    Path("data").write_text(data_output)
    Path("expected").write_text(expected_output)


if __name__ == "__main__":
    main()
