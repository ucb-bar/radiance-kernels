#include <stdint.h>


int main() {
  volatile int x = 1000;
  while (x--);
  return 0;
}
