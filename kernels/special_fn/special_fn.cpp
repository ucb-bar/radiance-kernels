#include <vx_intrinsics.h>
#include <math.h>

// TODO: write these
void measure_sin() {
    // sin of small angles (denormal)
    // sin of angles within [-2pi, 2pi]
    // sin of really big angles (forces reduction of angle)
}

void measure_cos() {
    // same test vectors as sin
}

void measure_exp() {
    // 
}

int main() {
  volatile float zero = 0.0f;
  volatile float cos_zero = cosf(zero);
  int cos_zero_i = *(int*)(&cos_zero);

  volatile int *cout = (int*)(0xFF080000);
  *cout = cos_zero_i;
  *cout = '\n';

  return 0;
}