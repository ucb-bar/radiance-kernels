#include "lib.h"

int main() {
  vx_tmc((1u << vx_num_threads()) - 1);

  exp();

  vx_tmc(1);

  return 0;
}
