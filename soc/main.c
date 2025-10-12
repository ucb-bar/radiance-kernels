// spin forever so we have a stable rv64 image
volatile unsigned long sink;
int main(void) {
  for (;;) { sink++; }
}

