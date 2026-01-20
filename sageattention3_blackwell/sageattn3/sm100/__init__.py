# SM100 (B200/B300) Blackwell Datacenter GPU support
# This module provides FP4 blockscaled attention using tcgen05.mma instructions
#
# Key architectural differences from SM120 (RTX 5090):
# - Uses tcgen05.mma instructions instead of mma.sync.aligned
# - Accumulators stored in TMEM (Tensor Memory) instead of registers
# - Supports flexible cluster shapes (SM120 is fixed 1x1x1)
# - 256KB TMEM per SM for tensor core accumulation
#
# The kernel is built with -arch=sm_100a flag
