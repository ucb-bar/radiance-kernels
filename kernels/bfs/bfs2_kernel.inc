#include <stdint.h>

#include <mu_intrinsics.h>
#include <mu_schedule.h>

struct Bfs2Args {
    __global uint32_t* graph_mask;
    __global uint32_t* updating_graph_mask;
    __global uint32_t* graph_visited;
    __global uint32_t* over;
    uint32_t no_of_nodes;
};

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
    Bfs2Args args{};
    args.graph_mask = graph_mask_raw;
    args.updating_graph_mask = updating_graph_mask_raw;
    args.graph_visited = graph_visited_raw;
    args.over = over_raw;
    args.no_of_nodes = no_of_nodes_val;
    mu_schedule(bfs2, &args, 4);
    return 0;
}
