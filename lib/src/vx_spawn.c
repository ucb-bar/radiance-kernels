// Copyright © 2019-2023
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NUM_CORES_MAX 1024

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

typedef struct {
	vx_spawn_tasks_cb callback;
	void* arg;
	int offset; // task offset
	int NWs;    // number of NW batches where NW=<total warps per core>.
	int RWs;    // number of remaining warps in the core
} wspawn_tasks_args_t;

typedef struct {
  context_t * ctx;
  vx_spawn_kernel_cb callback;
  void* arg;
	int offset; // task offset
	int NWs;    // number of NW batches where NW=<total warps per core>.
	int RWs;    // number of remaining warps in the core
  char isXYpow2;
  char log2XY;
  char log2X;
} wspawn_kernel_args_t;

void* g_wspawn_args[NUM_CORES_MAX];

inline char is_log2(int x) {
  return ((x & (x-1)) == 0);
}

inline int log2_fast(int x) {
  return 31 - __builtin_clz (x);
}

static void __attribute__ ((noinline)) spawn_tasks_all_stub() {
  int NT  = vx_num_threads();
  int cid = vx_core_id();
  int wid = vx_warp_id();
  int tid = vx_thread_id();

  wspawn_tasks_args_t* p_wspawn_args = (wspawn_tasks_args_t*)g_wspawn_args[cid];

  int wK = (p_wspawn_args->NWs * wid) + MIN(p_wspawn_args->RWs, wid);
  int tK = p_wspawn_args->NWs + (wid < p_wspawn_args->RWs);
  int offset = p_wspawn_args->offset + (wK * NT) + (tid * tK);

  vx_spawn_tasks_cb callback = p_wspawn_args->callback;
  void* arg = p_wspawn_args->arg;
  for (int task_id = offset, N = task_id + tK; task_id < N; ++task_id) {
    callback(task_id, arg);
  }
}

static void __attribute__ ((noinline)) spawn_tasks_contiguous_all_stub() {
  int NT  = vx_num_threads();
  int NW  = vx_num_warps();
  int cid = vx_core_id();
  int wid = vx_warp_id();
  int tid = vx_thread_id();

  wspawn_tasks_args_t* p_wspawn_args = (wspawn_tasks_args_t*)g_wspawn_args[cid];

  int waves = p_wspawn_args->NWs + (wid < p_wspawn_args->RWs);
  int offset = p_wspawn_args->offset + (NT * wid + tid);

  vx_spawn_tasks_cb callback = p_wspawn_args->callback;
  void* arg = p_wspawn_args->arg;
  for (int wave_id = 0; wave_id < waves; ++wave_id) {
    int task_id = offset + (wave_id * NT * NW);
    callback(task_id, arg);
  }
}

static void __attribute__ ((noinline)) spawn_tasks_cluster_all_stub() {
  int NT  = vx_num_threads();
  int NW  = vx_num_warps();
  int cid = vx_core_id();
  int wid = vx_warp_id();
  int tid = vx_thread_id();

  const int core_id_in_cluster = cid % CORES_PER_CLUSTER;
  // round-robin warp_id allocation across cores in cluster
  const int wid_in_cluster = CORES_PER_CLUSTER * wid + core_id_in_cluster;

  wspawn_tasks_args_t* p_wspawn_args = (wspawn_tasks_args_t*)g_wspawn_args[cid];

  int waves = p_wspawn_args->NWs + (wid < p_wspawn_args->RWs);
  int offset = p_wspawn_args->offset + (NT * wid_in_cluster + tid);

  vx_spawn_tasks_cb callback = p_wspawn_args->callback;
  void* arg = p_wspawn_args->arg;

  // sequential iterations
  for (int wave_id = 0; wave_id < waves; ++wave_id) {
    int task_id = offset + (wave_id * NT * NW * CORES_PER_CLUSTER);
    callback(task_id, arg);
  }
}

static void __attribute__ ((noinline)) spawn_tasks_rem_stub() {
  int cid = vx_core_id();
  int tid = vx_thread_id();
  
  wspawn_tasks_args_t* p_wspawn_args = (wspawn_tasks_args_t*)g_wspawn_args[cid];
  int task_id = p_wspawn_args->offset + tid;
  (p_wspawn_args->callback)(task_id, p_wspawn_args->arg);
}

static void __attribute__ ((noinline)) spawn_tasks_cluster_rem_stub() {
  int NT  = vx_num_threads();
  int cid = vx_core_id();
  int tid = vx_thread_id();
  int wid = vx_warp_id();

  const int core_id_in_cluster = cid % CORES_PER_CLUSTER;
  // round-robin warp_id allocation across cores in cluster
  const int wid_in_cluster = CORES_PER_CLUSTER * wid + core_id_in_cluster;

  wspawn_tasks_args_t* p_wspawn_args = (wspawn_tasks_args_t*)g_wspawn_args[cid];
  // FIXME: This assumes that all cores but the last one are working with full
  // warps, and only the last core has a partially-filled warp.
  int offset = p_wspawn_args->offset + (NT * wid_in_cluster + tid);

  int task_id = offset;
  (p_wspawn_args->callback)(task_id, p_wspawn_args->arg);
}

static void __attribute__ ((noinline)) spawn_tasks_contiguous_all_cb() {
  // activate all threads
  vx_tmc(-1);

  // call stub routine
  spawn_tasks_contiguous_all_stub();

  // disable warp
  vx_tmc_zero();
}

static void __attribute__ ((noinline)) spawn_tasks_cluster_all_cb() {
  // activate all threads
  vx_tmc(-1);

  // call stub routine
  spawn_tasks_cluster_all_stub();

  // disable warp
  vx_tmc_zero();
}

static void __attribute__ ((noinline)) spawn_tasks_all_cb() {
  // activate all threads
  vx_tmc(-1);

  // call stub routine
  spawn_tasks_all_stub();

  // disable warp
  vx_tmc_zero();
}

// This function runs in every core, but with only 1 warp and 1 thread enabled.
// The logic in this function figures out how many warps/threads this particular
// core has to enable to fulfill an entire grid of computation.
void vx_spawn_tasks_cluster(int num_tasks, vx_spawn_tasks_cb callback, void *arg) {
  // device specs
  const int NC = vx_num_cores();
  const int NW = vx_num_warps();
  const int NT = vx_num_threads();
  // NOTE: assumes divisible
  const int num_cluster = NC / CORES_PER_CLUSTER;

  // current core id
  int core_id = vx_core_id();
  if (core_id >= NUM_CORES_MAX)
    return;
  const int cluster_id = core_id / CORES_PER_CLUSTER;
  const int core_id_in_cluster = core_id % CORES_PER_CLUSTER;

  // try to fill up full clusters first
  const int num_threads_in_cluster = CORES_PER_CLUSTER * NW * NT;
  const int num_used_clusters =
      (num_tasks + (num_threads_in_cluster - 1)) / num_threads_in_cluster;
  if (cluster_id >= num_used_clusters) {
      return; // terminate extra clusters
  }
  // fill up the last cluster with remaining tasks
  const int num_full_clusters = num_tasks / num_threads_in_cluster;
  int num_tasks_this_cluster = num_threads_in_cluster;
  if (cluster_id >= num_full_clusters) {
      num_tasks_this_cluster = num_tasks % num_threads_in_cluster;
  }

  // Distribute threads equally across as many cores as possible, even if they
  // don't fill up NW*NT in a single core.  This makes sure the warps get evenly
  // distributed in a single cluster
  //
  // TODO: Try to contain in a single cluster if possible?
  const int num_active_cores = (num_tasks + (NT - 1)) / NT;
  if (core_id >= num_active_cores)
    return; // terminate extra cores

  const int num_full_warps_this_cluster = num_tasks_this_cluster / NT;
  const int rem_threads_in_last_warp = num_tasks_this_cluster % NT;
  // const int num_warps = (num_tasks_this_cluster + (NT - 1)) / NT;

  int num_warps_this_core = num_full_warps_this_cluster / CORES_PER_CLUSTER;
  const int num_warps_in_last_row = num_full_warps_this_cluster % CORES_PER_CLUSTER;
  if (core_id_in_cluster < num_warps_in_last_row) {
    num_warps_this_core++;
  }
  // if 0, last warp is full-threads enabled
  int rem_threads_in_last_warp_this_core = 0;
  if (rem_threads_in_last_warp != 0) {
    if (core_id_in_cluster == num_warps_in_last_row - 1) {
      rem_threads_in_last_warp_this_core = rem_threads_in_last_warp;
    }
  }

  // sequential iterations
  const int num_full_waves = num_warps_this_core / NW;
  const int rem_full_warps_in_last_wave = num_warps_this_core % NW;

  const const int offset = cluster_id * num_tasks_this_cluster;
  wspawn_tasks_args_t wspawn_args = {callback, arg, offset, num_full_waves,
                                     rem_full_warps_in_last_wave};
  g_wspawn_args[core_id] = &wspawn_args;

  if (num_warps_this_core > 0) {
    // execute callback on other warps
    const int nw = MIN(num_warps_this_core, NW);
    vx_wspawn(nw, spawn_tasks_cluster_all_cb);

    // activate all threads
    vx_tmc(-1);

    // call stub routine
    spawn_tasks_cluster_all_stub();

    // back to single-threaded
    vx_tmc_one();

    // wait for spawn warps to terminate
    vx_wspawn_wait();
  }

  // TODO: this is incomplete
  // TODO: Instead of launching an additional wave just to work on remaining
  // threads, handle this in the last wave amongst other full warps.
  if (rem_threads_in_last_warp != 0 && core_id_in_cluster == 0) {
    // adjust offset
    // FIXME: use rem_threads_in_last_warp_this_core
    wspawn_args.offset += (num_tasks_this_cluster - rem_threads_in_last_warp);

    // activate remaining threads
    const int tmask = (1 << rem_threads_in_last_warp) - 1;
    vx_tmc(tmask);

    // call stub routine
    spawn_tasks_cluster_rem_stub();

    // back to single-threaded
    vx_tmc_one();
  }
}

void vx_spawn_tasks_contiguous(int num_tasks, vx_spawn_tasks_cb callback , void * arg) {
	// device specs
  int NC = vx_num_cores();
  int NW = vx_num_warps();
  int NT = vx_num_threads();

  // current core id
  int core_id = vx_core_id();
  if (core_id >= NUM_CORES_MAX)
    return;

  // calculate necessary active cores
  int WT = NW * NT;
  int nC = (num_tasks > WT) ? (num_tasks / WT) : 1;
  int nc = MIN(nC, NC);
  if (core_id >= nc)
    return; // terminate extra cores

  // number of tasks per core
  int tasks_per_core = num_tasks / nc;
  int tasks_per_core_n1 = tasks_per_core;  
  if (core_id == (nc-1)) {    
    int rem = num_tasks - (nc * tasks_per_core); 
    tasks_per_core_n1 += rem; // last core also executes remaining tasks
  }

  // number of tasks per warp
  int TW = tasks_per_core_n1 / NT;      // occupied warps
  int rT = tasks_per_core_n1 - TW * NT; // remaining threads
  int fW = 1, rW = 0;
  if (TW >= NW) {
    fW = TW / NW;			                  // full warps iterations
    rW = TW - fW * NW;                  // remaining warps
  }

  wspawn_tasks_args_t wspawn_args = { callback, arg, core_id * tasks_per_core, fW, rW };
  g_wspawn_args[core_id] = &wspawn_args;

	if (TW >= 1)	{
    // execute callback on other warps
    int nw = MIN(TW, NW);
	  vx_wspawn(nw, spawn_tasks_contiguous_all_cb);

    // activate all threads
    vx_tmc(-1);

    // call stub routine
    spawn_tasks_contiguous_all_stub();
  
    // back to single-threaded
    vx_tmc_one();
    
    // wait for spawn warps to terminate
    vx_wspawn_wait();
	}  

  if (rT != 0) {
    // adjust offset
    wspawn_args.offset += (tasks_per_core_n1 - rT);
    
    // activate remaining threads  
    int tmask = (1 << rT) - 1;
    vx_tmc(tmask);

    // call stub routine
    spawn_tasks_rem_stub();

    // back to single-threaded
    vx_tmc_one();
  }
}

void vx_spawn_tasks(int num_tasks, vx_spawn_tasks_cb callback , void * arg) {
	// device specs
  int NC = vx_num_cores();
  int NW = vx_num_warps();
  int NT = vx_num_threads();

  // current core id
  int core_id = vx_core_id();
  if (core_id >= NUM_CORES_MAX)
    return;

  // calculate necessary active cores
  int WT = NW * NT;
  int nC = (num_tasks > WT) ? (num_tasks / WT) : 1;
  int nc = MIN(nC, NC);
  if (core_id >= nc)
    return; // terminate extra cores

  // number of tasks per core
  int tasks_per_core = num_tasks / nc;
  int tasks_per_core_n1 = tasks_per_core;  
  if (core_id == (nc-1)) {    
    int rem = num_tasks - (nc * tasks_per_core); 
    tasks_per_core_n1 += rem; // last core also executes remaining tasks
  }

  // number of tasks per warp
  int TW = tasks_per_core_n1 / NT;      // occupied warps
  int rT = tasks_per_core_n1 - TW * NT; // remaining threads
  int fW = 1, rW = 0;
  if (TW >= NW) {
    fW = TW / NW;			                  // full warps iterations
    rW = TW - fW * NW;                  // remaining warps
  }

  wspawn_tasks_args_t wspawn_args = { callback, arg, core_id * tasks_per_core, fW, rW };
  g_wspawn_args[core_id] = &wspawn_args;

	if (TW >= 1)	{
    // execute callback on other warps
    int nw = MIN(TW, NW);
	  vx_wspawn(nw, spawn_tasks_all_cb);

    // activate all threads
    vx_tmc(-1);

    // call stub routine
    spawn_tasks_all_stub();
  
    // back to single-threaded
    vx_tmc_one();
    
    // wait for spawn warps to terminate
    vx_wspawn_wait();
	}  

  if (rT != 0) {
    // adjust offset
    wspawn_args.offset += (tasks_per_core_n1 - rT);
    
    // activate remaining threads  
    int tmask = (1 << rT) - 1;
    vx_tmc(tmask);

    // call stub routine
    spawn_tasks_rem_stub();

    // back to single-threaded
    vx_tmc_one();
  }
}

///////////////////////////////////////////////////////////////////////////////

static void __attribute__ ((noinline)) spawn_kernel_all_stub() {
  int NT  = vx_num_threads();
  int cid = vx_core_id();
  int wid = vx_warp_id();
  int tid = vx_thread_id();

  wspawn_kernel_args_t* p_wspawn_args = (wspawn_kernel_args_t*)g_wspawn_args[cid];

  int wK = (p_wspawn_args->NWs * wid) + MIN(p_wspawn_args->RWs, wid);
  int tK = p_wspawn_args->NWs + (wid < p_wspawn_args->RWs);
  int offset = p_wspawn_args->offset + (wK * NT) + (tid * tK);

  int X = p_wspawn_args->ctx->num_groups[0];
  int Y = p_wspawn_args->ctx->num_groups[1];
  int XY = X * Y;

  if (p_wspawn_args->isXYpow2) {
    for (int wg_id = offset, N = wg_id + tK; wg_id < N; ++wg_id) {    
      int k = wg_id >> p_wspawn_args->log2XY;
      int wg_2d = wg_id - k * XY;
      int j = wg_2d >> p_wspawn_args->log2X;
      int i = wg_2d - j * X;
      (p_wspawn_args->callback)(p_wspawn_args->arg, p_wspawn_args->ctx, i, j, k);
    }
  } else {
    for (int wg_id = offset, N = wg_id + tK; wg_id < N; ++wg_id) {    
      int k = wg_id / XY;
      int wg_2d = wg_id - k * XY;
      int j = wg_2d / X;
      int i = wg_2d - j * X;
      (p_wspawn_args->callback)(p_wspawn_args->arg, p_wspawn_args->ctx, i, j, k);
    }
  }
}

static void __attribute__ ((noinline)) spawn_kernel_rem_stub() {
  int cid = vx_core_id();
  int tid = vx_thread_id();

  wspawn_kernel_args_t* p_wspawn_args = (wspawn_kernel_args_t*)g_wspawn_args[cid];

  int wg_id = p_wspawn_args->offset + tid;

  int X = p_wspawn_args->ctx->num_groups[0];
  int Y = p_wspawn_args->ctx->num_groups[1];
  int XY = X * Y;

  if (p_wspawn_args->isXYpow2) {
    int k = wg_id >> p_wspawn_args->log2XY;
    int wg_2d = wg_id - k * XY;
    int j = wg_2d >> p_wspawn_args->log2X;
    int i = wg_2d - j * X;
    (p_wspawn_args->callback)(p_wspawn_args->arg, p_wspawn_args->ctx, i, j, k);
  } else {
    int k = wg_id / XY;
    int wg_2d = wg_id - k * XY;
    int j = wg_2d / X;
    int i = wg_2d - j * X;
    (p_wspawn_args->callback)(p_wspawn_args->arg, p_wspawn_args->ctx, i, j, k);
  }
}

static void __attribute__ ((noinline)) spawn_kernel_all_cb() {  
  // activate all threads
  vx_tmc(-1);

  // call stub routine
  spawn_kernel_all_stub();

  // disable warp
  vx_tmc_zero();
}

void vx_spawn_kernel(context_t * ctx, vx_spawn_kernel_cb callback, void * arg) {  
  // total number of WGs
  int X  = ctx->num_groups[0];
  int Y  = ctx->num_groups[1];
  int Z  = ctx->num_groups[2];
  int XY = X * Y;
  int num_tasks = XY * Z;
  
  // device specs
  int NC = vx_num_cores();
  int NW = vx_num_warps();
  int NT = vx_num_threads();

  // current core id
  int core_id = vx_core_id();  
  if (core_id >= NUM_CORES_MAX)
    return;

  // calculate necessary active cores
  int WT = NW * NT;
  int nC = (num_tasks > WT) ? (num_tasks / WT) : 1;
  int nc = MIN(nC, NC);
  if (core_id >= nc)
    return; // terminate extra cores

  // number of tasks per core
  int tasks_per_core = num_tasks / nc;
  int tasks_per_core_n1 = tasks_per_core;  
  if (core_id == (nc-1)) {    
    int rem = num_tasks - (nc * tasks_per_core); 
    tasks_per_core_n1 += rem; // last core also executes remaining WGs
  }

  // number of tasks per warp
  int TW = tasks_per_core_n1 / NT;      // occupied warps
  int rT = tasks_per_core_n1 - TW * NT; // remaining threads
  int fW = 1, rW = 0;
  if (TW >= NW) {
    fW = TW / NW;			                  // full warps iterations
    rW = TW - fW * NW;                  // remaining warps
  }

  // fast path handling
  char isXYpow2 = is_log2(XY);
  char log2XY   = log2_fast(XY);
  char log2X    = log2_fast(X);

  wspawn_kernel_args_t wspawn_args = { 
    ctx, callback, arg, core_id * tasks_per_core, fW, rW, isXYpow2, log2XY, log2X
  };
  g_wspawn_args[core_id] = &wspawn_args;

	if (TW >= 1)	{
    // execute callback on other warps
    int nw = MIN(TW, NW);
	  vx_wspawn(nw, spawn_kernel_all_cb);

    // activate all threads
    vx_tmc(-1);

    // call stub routine
    asm volatile("" ::: "memory");
    spawn_kernel_all_stub();

    // back to single-threaded
    vx_tmc_one();
    
    // wait for spawn warps to terminate
    vx_wspawn_wait();
	}  

  if (rT != 0) {
    // adjust offset
    wspawn_args.offset += (tasks_per_core_n1 - rT);

    // activate remaining threads
    int tmask = (1 << rT) - 1;
    vx_tmc(tmask);

    // call stub routine
    spawn_kernel_rem_stub();

    // back to single-threaded
    vx_tmc_one();
  }
}

#ifdef __cplusplus
}
#endif
