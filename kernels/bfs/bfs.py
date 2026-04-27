#!/usr/bin/env python3

from pathlib import Path


NO_OF_NODES = 7
ADJACENCY = [
    [1, 2],
    [3, 4],
    [5, 6],
    [],
    [],
    [],
    [],
]


def to_u32(value: int) -> int:
    return value & 0xFFFFFFFF


def fmt_u32(values: list[int]) -> str:
    return ",".join(f"0x{to_u32(value):08x}" for value in values) + ","


def emit_array(name: str, values: list[int]) -> str:
    return f"""__global uint32_t {name}[] = {{
    {fmt_u32(values)}
}};
"""


def emit_const_array(name: str, values: list[int]) -> str:
    return f"""const uint32_t {name}[] = {{
    {fmt_u32(values)}
}};
"""


def emit_data(
    path: Path,
    graph_starting: list[int],
    graph_no_of_edges: list[int],
    graph_edges: list[int],
    graph_mask: list[int],
    updating_graph_mask: list[int],
    graph_visited: list[int],
    cost: list[int],
    over: list[int],
) -> None:
    text = (
        emit_array("graph_starting_raw", graph_starting)
        + "\n"
        + emit_array("graph_no_of_edges_raw", graph_no_of_edges)
        + "\n"
        + emit_array("graph_edges_raw", graph_edges)
        + "\n"
        + emit_array("graph_mask_raw", graph_mask)
        + "\n"
        + emit_array("updating_graph_mask_raw", updating_graph_mask)
        + "\n"
        + emit_array("graph_visited_raw", graph_visited)
        + "\n"
        + emit_array("cost_raw", cost)
        + "\n"
        + emit_array("over_raw", over)
        + "\n"
        + f"const uint32_t no_of_nodes_val = {NO_OF_NODES};\n"
        + f"const uint32_t edge_list_size_val = {len(graph_edges)};\n"
    )
    path.write_text(text)


def emit_expected(path: Path, arrays: dict[str, list[int]]) -> None:
    lines = []
    for name, values in arrays.items():
        lines.append(emit_const_array(name, values))
    path.write_text("\n".join(lines))


def emit_wrapper(path: Path, include_name: str, phase_macro: str) -> None:
    path.write_text(
        f'#define BFS_DATA_HEADER "{include_name}"\n'
        f"#define {phase_macro}\n"
        f'#include "kernel_impl.hpp"\n'
    )


def bfs1_step(
    graph_starting: list[int],
    graph_no_of_edges: list[int],
    graph_edges: list[int],
    graph_mask: list[int],
    updating_graph_mask: list[int],
    graph_visited: list[int],
    cost: list[int],
) -> None:
    for tid in range(NO_OF_NODES):
        if graph_mask[tid] == 0:
            continue
        graph_mask[tid] = 0
        start = graph_starting[tid]
        end = start + graph_no_of_edges[tid]
        next_cost = to_u32(cost[tid] + 1)
        for edge in range(start, end):
            node_id = graph_edges[edge]
            if graph_visited[node_id] == 0:
                cost[node_id] = next_cost
                updating_graph_mask[node_id] = 1


def bfs2_step(
    graph_mask: list[int],
    updating_graph_mask: list[int],
    graph_visited: list[int],
    over: list[int],
) -> None:
    over[0] = 0
    for tid in range(NO_OF_NODES):
        if updating_graph_mask[tid] == 0:
            continue
        graph_mask[tid] = 1
        graph_visited[tid] = 1
        over[0] = 1
        updating_graph_mask[tid] = 0


def remove_old_generated() -> None:
    patterns = (
        "bfs1_l*.cpp",
        "bfs2_l*.cpp",
        "bfs1_l*_data",
        "bfs2_l*_data",
        "bfs1_l*_expected",
        "bfs2_l*_expected",
    )
    for pattern in patterns:
        for path in Path(".").glob(pattern):
            path.unlink()


def build_graph() -> tuple[list[int], list[int], list[int]]:
    graph_starting = []
    graph_no_of_edges = []
    graph_edges = []
    cursor = 0
    for neighbors in ADJACENCY:
        graph_starting.append(cursor)
        graph_no_of_edges.append(len(neighbors))
        graph_edges.extend(neighbors)
        cursor += len(neighbors)
    return graph_starting, graph_no_of_edges, graph_edges


def clone_state(values: list[int]) -> list[int]:
    return values[:]


def main() -> None:
    remove_old_generated()

    graph_starting, graph_no_of_edges, graph_edges = build_graph()
    graph_mask = [1, 0, 0, 0, 0, 0, 0]
    updating_graph_mask = [0] * NO_OF_NODES
    graph_visited = [1, 0, 0, 0, 0, 0, 0]
    cost = [0] + [0xFFFFFFFF] * (NO_OF_NODES - 1)
    over = [0]

    level = 0
    while True:
        bfs1_stem = f"bfs1_l{level}"
        emit_data(
            Path(f"{bfs1_stem}_data"),
            graph_starting,
            graph_no_of_edges,
            graph_edges,
            graph_mask,
            updating_graph_mask,
            graph_visited,
            cost,
            over,
        )
        emit_wrapper(Path(f"{bfs1_stem}.cpp"), f"{bfs1_stem}_data", "BFS_KERNEL_BFS1")

        graph_mask_after_bfs1 = clone_state(graph_mask)
        updating_after_bfs1 = clone_state(updating_graph_mask)
        visited_after_bfs1 = clone_state(graph_visited)
        cost_after_bfs1 = clone_state(cost)
        bfs1_step(
            graph_starting,
            graph_no_of_edges,
            graph_edges,
            graph_mask_after_bfs1,
            updating_after_bfs1,
            visited_after_bfs1,
            cost_after_bfs1,
        )
        emit_expected(
            Path(f"{bfs1_stem}_expected"),
            {
                "expected_graph_mask_raw": graph_mask_after_bfs1,
                "expected_updating_graph_mask_raw": updating_after_bfs1,
                "expected_cost_raw": cost_after_bfs1,
            },
        )

        graph_mask = graph_mask_after_bfs1
        updating_graph_mask = updating_after_bfs1
        graph_visited = visited_after_bfs1
        cost = cost_after_bfs1

        bfs2_stem = f"bfs2_l{level}"
        over_before_bfs2 = [0]
        emit_data(
            Path(f"{bfs2_stem}_data"),
            graph_starting,
            graph_no_of_edges,
            graph_edges,
            graph_mask,
            updating_graph_mask,
            graph_visited,
            cost,
            over_before_bfs2,
        )
        emit_wrapper(Path(f"{bfs2_stem}.cpp"), f"{bfs2_stem}_data", "BFS_KERNEL_BFS2")

        graph_mask_after_bfs2 = clone_state(graph_mask)
        updating_after_bfs2 = clone_state(updating_graph_mask)
        visited_after_bfs2 = clone_state(graph_visited)
        cost_after_bfs2 = clone_state(cost)
        over_after_bfs2 = clone_state(over_before_bfs2)
        bfs2_step(
            graph_mask_after_bfs2,
            updating_after_bfs2,
            visited_after_bfs2,
            over_after_bfs2,
        )
        emit_expected(
            Path(f"{bfs2_stem}_expected"),
            {
                "expected_graph_mask_raw": graph_mask_after_bfs2,
                "expected_updating_graph_mask_raw": updating_after_bfs2,
                "expected_graph_visited_raw": visited_after_bfs2,
                "expected_over_raw": over_after_bfs2,
            },
        )

        graph_mask = graph_mask_after_bfs2
        updating_graph_mask = updating_after_bfs2
        graph_visited = visited_after_bfs2
        cost = cost_after_bfs2
        over = over_after_bfs2

        if over[0] == 0:
            break
        level += 1


if __name__ == "__main__":
    main()
