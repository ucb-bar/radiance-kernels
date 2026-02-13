#include <math.h>
#include <stdio.h>

static volatile int *cout = (int*)(0xFF080000);

// muon's libc doesn't define PI (??)
#define PI 3.14159265358979323846264338327950288419716939937510582f

static void puts(const char* str) {
    while (*str != '\0') {
        *cout = *str++;
    }
    *cout = '\n';
}

// muon's libc doesn't define PI (??)
#define PI 3.14159265358979323846264338327950288419716939937510582f
#define PI_2 1.5707963267948966192313216916397514420985846996875529f

#define EPS 1.0e-6

// special functions don't always give the same answer across platforms
static void assert_close(float a, float b) {
  char buf[128];
  if (fabsf(a - b) > EPS) {
      puts("bad");
  }
}

volatile float sin_test_vector[] = { 
  #include "sin"
};

void correctness_sin() {
  #define CHECK_SIN(arg, expected) {      \
      float input = (arg);                \
      volatile float x = sinf(input);     \
      assert_close(x, expected);          \
  }

  
  unsigned int length = sizeof(sin_test_vector) / sizeof(float);

  for (int i = 0; i < length / 2; i += 1) {
      CHECK_SIN(sin_test_vector[i*2], sin_test_vector[i*2+1]);
  }
}

void measure_sin() {
  volatile float input = PI / 17.0f;
  volatile float x;
  for (int i = 0; i < 1000; i += 1) {
    x = sinf(input);
  }
  asm("");
}

int main() {
  correctness_sin();
  measure_sin();
  return 0;
}