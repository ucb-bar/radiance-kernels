#include <math.h>
#include <stdio.h>
#include <string.h>
#include <mu_intrinsics.h>

static volatile int *cout = (int*)(0xFF080000);

// muon's libc doesn't define PI (??)
#define PI 3.14159265358979323846264338327950288419716939937510582f

static void puts(const char* str) {
    while (*str != '\0') {
        *cout = *str++;
    }
    *cout = '\n';
}

// can't use bit_cast since not C++20
static int float_to_hex(float x) {
  int y;
  memcpy(&y, &x, sizeof(float));
  return y;
}

// special functions don't always give the same answer across platforms
static void assert_close(float a, float b, float rtol = 1.0e-5f, float atol = 1.0e-6f) {
  char buf[128];
  if (fabsf(a - b) > (atol + rtol * fabsf(b))) {
      sprintf(
        buf, 
        "bad: a = 0x%.8x, b = 0x%.8x", 
        float_to_hex(a), 
        float_to_hex(b)
      );
      
      puts(buf);
  }
}

volatile float sin_test_vector[] = { 
  #include "sin"
};

volatile float cos_test_vector[] = {
  #include "cos"
};

__attribute__((noinline)) void correctness_sin() {
  unsigned int length = sizeof(sin_test_vector) / sizeof(float);

  for (int i = 0; i < length / 2; i += 1) {
    float input = sin_test_vector[i*2];
    volatile float x = sinf(input);
    assert_close(x, sin_test_vector[i*2+1]);
  }
}

__attribute__((noinline)) void measure_sin() {
  volatile float input = PI / 17.0f;
  volatile float x;
  
  // generated code does 1 load, 1 store per sinf, which i think is fair
  for (int i = 0; i < 1000; i += 1) {
    x = sinf(input);
  }
}

__attribute__((noinline)) void correctness_cos() {
  unsigned int length = sizeof(cos_test_vector) / sizeof(float);

  for (int i = 0; i < length / 2; i += 1) {
    float input = cos_test_vector[i*2];
    volatile float x = cosf(input);
    assert_close(x, cos_test_vector[i*2+1]);
  }
}

__attribute__((noinline)) void measure_cos() {
  volatile float input = PI / 17.0f;
  volatile float x;
  
  // generated code does 1 load, 1 store per cosf, which i think is fair
  for (int i = 0; i < 1000; i += 1) {
    x = cosf(input);
  }
}

volatile float exp_test_vector[] = {
  #include "exp"
};

__attribute__((noinline)) void correctness_exp() {
  unsigned int length = sizeof(exp_test_vector) / sizeof(float);

  for (int i = 0; i < length / 2; i += 1) {
    float input = exp_test_vector[i*2];
    volatile float x = mu_fexp(input);
    assert_close(x, exp_test_vector[i*2+1]);
  }
}

int main() {
  correctness_sin();
  correctness_cos();

  return 0;
}