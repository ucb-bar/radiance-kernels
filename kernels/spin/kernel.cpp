int main() {
    asm volatile ("1:\nj 1b\n");
    return 0;
}
