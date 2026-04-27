#include <stdint.h>

#include <mu_intrinsics.h>
#include <mu_schedule.h>

#ifndef BFS_NUM_WARPS
#define BFS_NUM_WARPS 4
#endif

#ifndef BFS_DATA_HEADER
#error "BFS_DATA_HEADER must be defined before including kernel_impl.hpp"
#endif

extern "C" uint32_t __mu_num_warps = BFS_NUM_WARPS;

struct Bfs1Args {
    __global uint32_t* graph_starting;
    __global uint32_t* graph_no_of_edges;
    __global uint32_t* graph_edges;
    __global uint32_t* graph_mask;
    __global uint32_t* updating_graph_mask;
    __global uint32_t* graph_visited;
    __global uint32_t* cost;
    uint32_t no_of_nodes;
};

struct Bfs2Args {
    __global uint32_t* graph_mask;
    __global uint32_t* updating_graph_mask;
    __global uint32_t* graph_visited;
    __global uint32_t* over;
    uint32_t no_of_nodes;
};

static void bfs1(void* raw_arg, uint32_t tid_in_threadblock,
                 uint32_t threads_per_threadblock, uint32_t) {
    auto* arg = reinterpret_cast<Bfs1Args*>(raw_arg);
    for (uint32_t tid = tid_in_threadblock; tid < arg->no_of_nodes;
         tid += threads_per_threadblock) {
        if (arg->graph_mask[tid] == 0) {
            continue;
        }

        arg->graph_mask[tid] = 0;
        uint32_t start = arg->graph_starting[tid];
        uint32_t end = start + arg->graph_no_of_edges[tid];
        uint32_t next_cost = arg->cost[tid] + 1;
        for (uint32_t edge = start; edge < end; ++edge) {
            uint32_t id = arg->graph_edges[edge];
            if (arg->graph_visited[id] == 0) {
                arg->cost[id] = next_cost;
                arg->updating_graph_mask[id] = 1;
            }
        }
    }
}

static void bfs2(void* raw_arg, uint32_t tid_in_threadblock,
                 uint32_t threads_per_threadblock, uint32_t) {
    auto* arg = reinterpret_cast<Bfs2Args*>(raw_arg);
    for (uint32_t tid = tid_in_threadblock; tid < arg->no_of_nodes;
         tid += threads_per_threadblock) {
        if (arg->updating_graph_mask[tid] == 0) {
            continue;
        }

        arg->graph_mask[tid] = 1;
        arg->graph_visited[tid] = 1;
        arg->over[0] = 1;
        arg->updating_graph_mask[tid] = 0;
    }
}

#include BFS_DATA_HEADER

int main() {
#if defined(BFS_KERNEL_BFS1)
    Bfs1Args args{};
    args.graph_starting = graph_starting_raw;
    args.graph_no_of_edges = graph_no_of_edges_raw;
    args.graph_edges = graph_edges_raw;
    args.graph_mask = graph_mask_raw;
    args.updating_graph_mask = updating_graph_mask_raw;
    args.graph_visited = graph_visited_raw;
    args.cost = cost_raw;
    args.no_of_nodes = no_of_nodes_val;
    mu_schedule(bfs1, &args, BFS_NUM_WARPS);
#elif defined(BFS_KERNEL_BFS2)
    Bfs2Args args{};
    args.graph_mask = graph_mask_raw;
    args.updating_graph_mask = updating_graph_mask_raw;
    args.graph_visited = graph_visited_raw;
    args.over = over_raw;
    args.no_of_nodes = no_of_nodes_val;
    mu_schedule(bfs2, &args, BFS_NUM_WARPS);
#else
#error "Define BFS_KERNEL_BFS1 or BFS_KERNEL_BFS2 before including kernel_impl.hpp"
#endif
    return 0;
}
