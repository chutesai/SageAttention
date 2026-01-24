/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SM100 (B200/B300) FP4 Block-Scaled kernel traits for SageAttention3.
 *
 * This uses SM100's tcgen05.mma block-scaled FP4 instructions (SM100_MMA_MXF4_SS).
 *
 * Key Design Decisions:
 * ---------------------
 * 1. SM100 FP4 MMA uses K=64 elements (256 bits) per MMA operation
 * 2. We do NOT use CollectiveBuilder since it's designed for GEMM, not FMHA
 * 3. Instead, we follow the SM120 pattern of directly constructing TiledMma
 * 4. Block-scaled format: E2M1 data with E4M3 scale factors, 16 elements per SF
 *
 * For Flash Attention:
 * - QK GEMM: Q(BlockM×HeadDim) @ K^T(HeadDim×BlockN) → S(BlockM×BlockN)
 * - PV GEMM: P(BlockM×BlockN) @ V(BlockN×HeadDim) → O(BlockM×HeadDim)
 */

#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

// Include SM100 MMA atoms
#include "cute/arch/mma_sm100.hpp"
#include "cute/atom/mma_traits_sm100.hpp"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// SM100 FP4 Block-Scaled Flash Forward Kernel Traits
//
// This follows the pattern from SM120 kernel_traits.h but adapts for SM100
// by using SM100_MMA_MXF4_SS instead of SM120's mma.sync atoms
///////////////////////////////////////////////////////////////////////////////

template <
    int kHeadDim_,    // Head dimension (128 or 256)
    int kBlockM_,     // Block size for Q rows (128 or 256)
    int kBlockN_,     // Block size for K/V sequence (128 or 256)
    int kStages_,     // Pipeline stages (2-4)
    int kClusterM_,   // Cluster shape in M (1)
    bool BlockMean_,  // Whether to use block mean subtraction
    typename ElementOut_ = cutlass::bfloat16_t
>
struct Flash_fwd_kernel_traits_sm100_fp4 {

    // Basic configuration
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr bool BlockMean = BlockMean_;
    static constexpr int kClusterM = kClusterM_;
    static constexpr int kStages = kStages_;
    static constexpr int EpiStages = 2;
    // For simplified scalar implementation: 1 thread per row
    // For full tensor core implementation: 16 warps
    static constexpr int kNWarps = kBlockM / 32;  // Each warp handles 32 rows
    static constexpr int kNThreads = kBlockM;      // 1 thread per row for scalar impl

    //=========================================================================
    // Element Types for FP4 Block-Scaled Attention
    //=========================================================================

    // FP4 format: E2M1 data with E4M3 scale factors
    using Element = cutlass::float_e2m1_t;           // FP4 data (E2M1)
    using ElementSF = cutlass::float_e4m3_t;         // Scale factor (E4M3)
    using ElementAccum = float;                       // FP32 accumulator
    using ElementOut = ElementOut_;                   // Output type (BF16)
    using index_t = int64_t;

    // Scale factor configuration - 16 elements per scale factor block
    static constexpr int SFVectorSize = 16;
    static constexpr int NumSFPerHeadDim = kHeadDim / SFVectorSize;
    static constexpr int NumSFPerBlockN = kBlockN / SFVectorSize;

    //=========================================================================
    // Tile Shapes
    //=========================================================================

    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape_MNK = Shape<Int<kClusterM>, _1, _1>;

    //=========================================================================
    // MMA Tile Configuration
    //=========================================================================

    // SM100_MMA_MXF4_SS has fixed M=128, so:
    // - For kBlockM=128: 1 atom along M
    // - For kBlockM=256: 2 atoms along M (requires multi-SM cluster or iteration)
    //
    // The N dimension of the MMA is configurable (8-256, multiple of 8)
    // We use N=128 to match kBlockN

    // For kBlockM=128, one MMA atom covers the entire M dimension
    // For kBlockM=256, we need 2 atoms along M
    static constexpr int NumMmaAtomsM = kBlockM / 128;
    static constexpr int NumMmaAtomsN = kBlockN / 128;  // N=128 per atom

    using AtomLayoutMNK = Layout<Shape<Int<NumMmaAtomsM>, Int<NumMmaAtomsN>, _1>>;

    // The permutation tile describes how to tile the compute over the MMA atoms
    using PermTileM = _128;  // MMA atom M size
    using PermTileN = _128;  // MMA atom N size
    using PermTileK = Int<kHeadDim>;

    //=========================================================================
    // TiledMma Definitions - Direct construction like SM120
    //
    // SM100 uses SM100_MMA_MXF4_SS for FP4 block-scaled MMA
    // The MMA atom shape is (M, N, K) where K=64 for FP4 (256 bits / 4 bits)
    //
    // NOTE: These are defined but not currently used in the simplified
    // scalar mainloop. Full tensor core implementation will use these.
    //=========================================================================

    // For QK GEMM: A=Q (row-major), B=K^T (col-major, i.e., K transposed)
    // SM100_MMA_MXF4_SS<a_type, b_type, c_type, sf_type, M, N, VS, a_major, b_major, a_neg, b_neg>
    // where:
    //   - M, N are the MMA tile dimensions (M=128 fixed, N=8-256)
    //   - VS = SFVectorSize (16 or 32)
    //   - a_major = UMMA::Major::K (row-major A)
    //   - b_major = UMMA::Major::K (col-major B)
    //   - a_neg, b_neg = UMMA::ScaleIn::One (no negation)

    // Placeholder types for future tensor core implementation
    // The actual MMA will be:
    // SM100_MMA_MXF4_SS<Element, Element, ElementAccum, ElementSF,
    //                   128, 128, SFVectorSize,
    //                   UMMA::Major::K, UMMA::Major::K,
    //                   UMMA::ScaleIn::One, UMMA::ScaleIn::One>
    using TiledMmaQK = void;  // Placeholder - will be properly defined for tensor core impl
    using TiledMmaPV = void;  // Placeholder - will be properly defined for tensor core impl

    //=========================================================================
    // Scale Factor Configuration
    //=========================================================================

    using BlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVectorSize>;
    using LayoutSF = decltype(BlkScaledConfig::deduce_layoutSFA());

    static constexpr int MMA_NSF = 64 / SFVectorSize;  // MMA K=64, SF covers 16 elements

    //=========================================================================
    // NOTE: Full tensor core implementation will need:
    // - TMA Copy Atoms (SM90_TMA_LOAD)
    // - SMEM Layouts for Q/K/V/O
    // - Scale Factor SMEM Layouts
    // - Copy Atoms for SMEM access
    //
    // These are commented out for the simplified scalar implementation
    // to avoid issues with sm100_smem_selector and FP4 types.
    //=========================================================================

    //=========================================================================
    // Shared Storage
    //=========================================================================

    // Simplified shared storage for scalar fallback implementation
    // Full tensor core implementation will need proper SMEM buffers
    struct SharedStorage {
        // Minimal storage - scalar implementation reads from global memory
        // Just need some scratch space for synchronization
        alignas(128) char scratch[256];
    };

    //=========================================================================
    // NOTE: Pipeline types and TMA transaction sizes are commented out
    // for the simplified scalar implementation.
    // Full tensor core implementation will need these.
    //=========================================================================

    //=========================================================================
    // Strides (used for global memory access)
    //=========================================================================

    using StrideQ = Stride<int64_t, _1, int64_t>;
    using StrideK = Stride<int64_t, _1, int64_t>;
    using StrideV = Stride<_1, int64_t, int64_t>;  // V is transposed (dim, seq, batch*head)
    using StrideO = Stride<int64_t, _1, int64_t>;
};

///////////////////////////////////////////////////////////////////////////////
// SM100 FP4 Block-Scaled Flash Forward Kernel Traits - Tensor Core Variant
//
// This version has proper SMEM sizing for cooperative data loading
// and tensor core execution (hybrid implementation).
///////////////////////////////////////////////////////////////////////////////

template <
    int kHeadDim_,    // Head dimension (128 or 256)
    int kBlockM_,     // Block size for Q rows (128 or 256)
    int kBlockN_,     // Block size for K/V sequence (128 or 256)
    int kStages_,     // Pipeline stages (2-4)
    int kClusterM_,   // Cluster shape in M (1)
    bool BlockMean_,  // Whether to use block mean subtraction
    typename ElementOut_ = cutlass::bfloat16_t
>
struct Flash_fwd_kernel_traits_sm100_fp4_tc {

    // Basic configuration
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr bool BlockMean = BlockMean_;
    static constexpr int kClusterM = kClusterM_;
    static constexpr int kStages = kStages_;
    static constexpr int EpiStages = 2;

    // For tensor core implementation with SMEM staging
    // Use 4 warps (128 threads) for better occupancy
    static constexpr int kNWarps = 4;
    static constexpr int kNThreads = kNWarps * 32;  // 128 threads

    //=========================================================================
    // Element Types for FP4 Block-Scaled Attention
    //=========================================================================

    using Element = cutlass::float_e2m1_t;           // FP4 data (E2M1)
    using ElementSF = cutlass::float_e4m3_t;         // Scale factor (E4M3)
    using ElementAccum = float;                       // FP32 accumulator
    using ElementOut = ElementOut_;                   // Output type (BF16)
    using index_t = int64_t;

    // Scale factor configuration - 16 elements per scale factor block
    static constexpr int SFVectorSize = 16;
    static constexpr int NumSFPerHeadDim = kHeadDim / SFVectorSize;
    static constexpr int NumSFPerBlockN = kBlockN / SFVectorSize;

    //=========================================================================
    // Tile Shapes
    //=========================================================================

    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape_MNK = Shape<Int<kClusterM>, _1, _1>;

    //=========================================================================
    // MMA Configuration
    //=========================================================================

    // SM100_MMA_MXF4_SS: M=128, N=8-256, K=64
    static constexpr int kMmaM = 128;
    static constexpr int kMmaN = 128;
    static constexpr int kMmaK = 64;

    static constexpr int NumMmaAtomsM = kBlockM / kMmaM;
    static constexpr int NumMmaAtomsN = kBlockN / kMmaN;

    //=========================================================================
    // Scale Factor Configuration
    //=========================================================================

    using BlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVectorSize>;
    using LayoutSF = decltype(BlkScaledConfig::deduce_layoutSFA());

    static constexpr int MMA_NSF = kMmaK / SFVectorSize;

    //=========================================================================
    // SMEM Sizes (in bytes)
    //=========================================================================

    // FP4 data is packed (2 values per byte)
    static constexpr int SmemSizeQ = kBlockM * kHeadDim / 2;
    static constexpr int SmemSizeK = kBlockN * kHeadDim / 2;
    static constexpr int SmemSizeV = kBlockN * kHeadDim / 2;

    // Scale factors (1 byte per FP8 E4M3)
    static constexpr int SmemSizeSFQ = kBlockM * NumSFPerHeadDim;
    static constexpr int SmemSizeSFK = kBlockN * NumSFPerHeadDim;
    static constexpr int SmemSizeSFV = kBlockN * NumSFPerHeadDim;

    // Scratch space for softmax statistics
    static constexpr int SmemSizeScratch = kBlockM * 4 * sizeof(float);

    // Total SMEM (with 128-byte alignment)
    static constexpr int SmemSizeTotal =
        ((SmemSizeQ + 127) / 128 * 128) +
        ((SmemSizeK + 127) / 128 * 128) +
        ((SmemSizeV + 127) / 128 * 128) +
        ((SmemSizeSFQ + 127) / 128 * 128) +
        ((SmemSizeSFK + 127) / 128 * 128) +
        ((SmemSizeSFV + 127) / 128 * 128) +
        ((SmemSizeScratch + 127) / 128 * 128);

    //=========================================================================
    // Shared Storage for Tensor Core Implementation
    //=========================================================================

    struct SharedStorage {
        // FP4 data tiles (packed, 2 values per byte)
        alignas(128) uint8_t smem_Q[SmemSizeQ];
        alignas(128) uint8_t smem_K[SmemSizeK];
        alignas(128) uint8_t smem_V[SmemSizeV];

        // Scale factor tiles
        alignas(128) ElementSF smem_SFQ[kBlockM * NumSFPerHeadDim];
        alignas(128) ElementSF smem_SFK[kBlockN * NumSFPerHeadDim];
        alignas(128) ElementSF smem_SFV[kBlockN * NumSFPerHeadDim];

        // Scratch for softmax stats
        alignas(128) float smem_scratch[kBlockM * 4];
    };

    //=========================================================================
    // Strides
    //=========================================================================

    using StrideQ = Stride<int64_t, _1, int64_t>;
    using StrideK = Stride<int64_t, _1, int64_t>;
    using StrideV = Stride<_1, int64_t, int64_t>;
    using StrideO = Stride<int64_t, _1, int64_t>;
};

} // namespace flash
