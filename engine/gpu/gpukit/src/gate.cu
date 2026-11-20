// The kernel gate.rs launches.

// Spins until the host sets the flag, then clears it.
extern "C" __global__ void tessera_hold(volatile unsigned* flag, unsigned spins) {
  for (unsigned i = 0; i < spins && *flag == 0; i++) {
  }
  *flag = 0;
}
