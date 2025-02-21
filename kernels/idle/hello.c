#include <stdio.h>
#include <stdbool.h>

int main() {
    int value32 = 4097;  // Value for x32
    int value50 = -16778000;  // Value for x50
    int value71 = 2147483647;  // Value for x71
    int value125 = -2147483648; // Value for x125
    int sum;

    asm volatile (
        // Assign values to the registers
        "mv x32, %1\n"
        "mv x50, %2\n"
        "mv x71, %3\n"
        "mv x125, %4\n"

        // Sum the values in the registers
        "add x32, x32, x50\n"  // x32 = x32 + x50
        "add x32, x32, x71\n"  // x32 = x32 + x71
        "add x32, x32, x125\n" // x32 = x32 + x125

        // Move the result back to a C variable
        "mv %0, x32\n"
        : "=r" (sum) // Output
        : "r" (value32), "r" (value50), "r" (value71), "r" (value125) // Inputs
        : "x32", "x50", "x71", "x125" // Clobbered registers
    );
    volatile bool test = *(&value125 + 4);
    if (test) {
        printf("hello world 1!");
    } else {
        printf("hello world 2!");
    }

    // asm volatile (".insn r5 0, 0, 0, x101, x102, x103, x104, x105");

    return sum;
}
