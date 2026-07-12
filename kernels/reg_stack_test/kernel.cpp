volatile int g_sink = 0;

static inline int mix(int x) {
  int a0 = x + 1;
  int a1 = x + 2;
  int a2 = x + 3;
  int a3 = x + 4;
  int a4 = x + 5;
  int a5 = x + 6;
  int a6 = x + 7;
  int a7 = x + 8;
  int acc = 0;
  acc += a0 + a1 + a2 + a3;
  acc += a4 + a5 + a6 + a7;
  return acc;
}

int main() {
  volatile int slot0 = 11;
  volatile int slot1 = 22;
  volatile int slot2 = 33;
  int v = mix(slot0 + slot1 + slot2);
  g_sink = v + slot0 + slot1 + slot2;
  return g_sink;
}
