#include <stdint.h>
#include <vx_intrinsics.h>
#include <mu_schedule.h>

struct SoftmaxArgs {
  float* input;
  float* output;
  uint32_t rows;
  uint32_t cols;
};

void softmax(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args = reinterpret_cast<SoftmaxArgs*>(arg);

}

SoftmaxArgs softmax_args = {
  .input = nullptr,
  .output = nullptr,
  .rows = 0,
  .cols = 0,
};

int main() {
  mu_schedule(softmax, &softmax_args);
  return 0;
}
