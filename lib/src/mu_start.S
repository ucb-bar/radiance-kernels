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

// #include <VX_config.h>
// #include <VX_types.h>

#define RISCV_CUSTOM0   0x0B

#define STACK_BASE_ADDR 0x7F000000
#define STACK_LOG2_SIZE 13
#define CSR_WMASK 0xCC3

.section .init, "ax"
.global _start
.type   _start, @function
_start:  

  # initialize per-thread registers
  csrr  t0, nw
  la    t1, init_regs_all
  .insn r RISCV_CUSTOM0, 1, 0, x0, t0, t1  # wspawn t0, t1
  li    t0, -1
  vx_tmc t0
  jal   init_regs
  li    t0, 1
  vx_tmc t0
#   .insn r RISCV_CUSTOM0, 0, 0, x0, t0, x0  # tmc t0

  # wait for spawn warps to terminate
  # jal vx_wspawn_wait

  # clear BSS segment
  # TODO: re-enable for global/statics
  # la    a0, _edata
  # la    a2, _end
  # sub   a2, a2, a0
  # li    a1, 0
  # call  memset

  # run global initialization functions
  # TODO: re-enable with libc
  # call  __libc_init_array

  # call main program routine
  call main

  # call exit routine
  # TODO: re-enable with libc
  # tail  exit
  tail _exit
.size _start, .-_start

.section .text
.type _exit, @function
.global _exit
_exit:
  .insn r RISCV_CUSTOM0, 0, 0, x0, x0, x0  # tmc x0

.section .text
.type init_regs, @function
.local init_regs
init_regs:
  # set global pointer register
  .option push
  .option norelax
  la    gp, __global_pointer
  .option pop
  
  # set stack pointer register
  li    sp, STACK_BASE_ADDR # load stack base address
  csrr  t0, mhartid
  sll   t1, t0, STACK_LOG2_SIZE
  sll   t2, t0, 4
  add   t1, t1, t2
  sub   sp, sp, t1

  # set thread pointer register
  # use address space after BSS region
  # ensure cache line alignment
  la    t1, __tcb_aligned_size
  mul   t0, t0, t1
  la    tp, _end + 63
  add   tp, tp, t0
  and   tp, tp, -64
  ret

.section .text
.type init_regs_all, @function
.local init_regs_all
init_regs_all:
  li    t0, -1
  .insn r RISCV_CUSTOM0, 0, 0, x0, t0, x0  # tmc t0
  jal   init_regs
  .insn r RISCV_CUSTOM0, 0, 0, x0, x0, x0  # tmc x0
  ret  

.section .text
.type vx_wspawn_wait, @function
.global vx_wspawn_wait
vx_wspawn_wait:
  csrr  t0, CSR_WMASK
  li    t1, 1
  bne   t0, t1, vx_wspawn_wait
  ret

.section .data
	.global __dso_handle
	.weak __dso_handle
__dso_handle:
	.long	0

