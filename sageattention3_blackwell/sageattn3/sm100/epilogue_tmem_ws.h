/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SM100 (B200/B300) Epilogue for FlashAttention.
 *
 * KEY DIFFERENCES FROM SM120:
 * ==========================
 * 1. Output accumulator comes from mainloop as raw float array
 * 2. Must convert float to bf16/fp16 and write to SMEM
 * 3. Then use TMA to store SMEM to GMEM
 */

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "../blackwell/named_barrier.h"
#include "../blackwell/utils.h"

namespace flash {

using namespace cute;

template <typename Ktraits>
struct CollectiveEpilogueFwdSm100 {

    using Element = typename Ktraits::ElementOut;
    using ElementAccum = typename Ktraits::ElementAccum;
    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockN = Ktraits::kBlockN;
    static constexpr int kHeadDim = Ktraits::kHeadDim;
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    static constexpr int kNWarps = Ktraits::kNWarps;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int NumMmaThreads = kNThreads - cutlass::NumThreadsPerWarpGroup;

    using GmemTiledCopyOTMA = cute::SM90_TMA_STORE;

    // Output SMEM layout
    using SmemLayoutO = typename Ktraits::SmemLayoutO;

    // Non-TMA output store for special cases (e.g., writing zeros)
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kHeadDim / kGmemElemsPerLoad;
    static_assert(NumMmaThreads % kGmemThreadsPerRow == 0, "NumMmaThreads must be a multiple of kGmemThreadsPerRow");

    using GmemLayoutAtom = Layout<Shape<Int<NumMmaThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));

    using SmemCopyAtomO = Copy_Atom<SM90_U32x2_STSM_N, Element>;
    using SharedStorage = typename Ktraits::SharedStorage;

    using ShapeO = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen_q, d, head, batch)
    using StrideO = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using StrideLSE = cute::Stride<_1, int64_t, int64_t>;           // (seqlen_q, head, batch)

    using TMA_O = decltype(make_tma_copy(
        GmemTiledCopyOTMA{},
        make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)),
                    repeat_like(StrideO{}, int32_t(0)), StrideO{}),
        SmemLayoutO{},
        select<0, 2>(TileShape_MNK{}),
        _1{}));  // no mcast for O

    // Host side kernel arguments
    struct Arguments {
        Element* ptr_O;
        ShapeO const shape_O;
        StrideO const stride_O;
        float* ptr_LSE;
        StrideLSE const stride_LSE;
    };

    // Device side kernel params
    struct Params {
        Element* ptr_O;
        ShapeO const shape_O;
        StrideO const stride_O;
        float* ptr_LSE;
        StrideLSE const stride_LSE;
        TMA_O tma_store_O;
    };

    static Params to_underlying_arguments(Arguments const& args) {
        Tensor mO = make_tensor(make_gmem_ptr(args.ptr_O), args.shape_O, args.stride_O);
        TMA_O tma_store_O = make_tma_copy(
            GmemTiledCopyOTMA{},
            mO,
            SmemLayoutO{},
            select<0, 2>(TileShape_MNK{}),
            _1{});
        return {args.ptr_O, args.shape_O, args.stride_O, args.ptr_LSE, args.stride_LSE, tma_store_O};
    }

    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& epilogue_params) {
        cute::prefetch_tma_descriptor(epilogue_params.tma_store_O.get_tma_descriptor());
    }

    ///////////////////////////////////////////////////////////////////////////
    // Store output from register fragment to SMEM, then TMA to GMEM
    // This version works with cute tensor fragments
    ///////////////////////////////////////////////////////////////////////////
    template <typename FrgTensorO, typename TiledMma>
    CUTLASS_DEVICE void
    mma_store(
        SharedStorage& shared_storage,
        TiledMma tiled_mma,
        FrgTensorO const& tOrO,
        int thread_idx
    ) {
        Tensor sO = cute::as_position_independent_swizzle_tensor(
            make_tensor(make_smem_ptr(shared_storage.smem_o.data()), SmemLayoutO{}));

        // Convert float accumulator to output type (bf16/fp16)
        constexpr int numel = decltype(size(tOrO))::value;
        cutlass::NumericArrayConverter<Element, float, numel> convert_op;
        auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, numel>*>(tOrO.data()));
        auto tOrO_out = make_tensor(make_rmem_ptr<Element>(&frag), tOrO.layout());

        // Copy from registers to SMEM
        auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
        auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);
        Tensor taccOrO = smem_thr_copy_O.retile_S(tOrO_out);
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
        cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
        cutlass::arch::fence_view_async_shared();  // ensure smem writes are visible to TMA
    }

    ///////////////////////////////////////////////////////////////////////////
    // Store output from raw float array to SMEM
    // This version is for SM100 where output is in raw float arrays
    ///////////////////////////////////////////////////////////////////////////
    CUTLASS_DEVICE void
    store_from_array(
        SharedStorage& shared_storage,
        float const* acc_o,
        int num_elems,
        int thread_idx
    ) {
        Element* smem_o = shared_storage.smem_o.data();

        // Each thread writes its portion
        int const elems_per_thread = num_elems;
        int const start_elem = thread_idx * elems_per_thread;

        // Convert and store to SMEM
        // Process in chunks for better memory coalescing
        #pragma unroll
        for (int i = 0; i < elems_per_thread; i += 4) {
            // Convert 4 floats to bf16/fp16
            cutlass::NumericArrayConverter<Element, float, 4> convert_op;
            cutlass::Array<float, 4> src_arr;
            src_arr[0] = (i + 0 < elems_per_thread) ? acc_o[i + 0] : 0.0f;
            src_arr[1] = (i + 1 < elems_per_thread) ? acc_o[i + 1] : 0.0f;
            src_arr[2] = (i + 2 < elems_per_thread) ? acc_o[i + 2] : 0.0f;
            src_arr[3] = (i + 3 < elems_per_thread) ? acc_o[i + 3] : 0.0f;

            auto dst_arr = convert_op(src_arr);

            // Calculate SMEM index
            int global_idx = start_elem + i;
            int row = global_idx / kHeadDim;
            int col = global_idx % kHeadDim;

            // Write to SMEM with proper layout
            // SmemLayoutO is (kBlockM, kHeadDim) with swizzling
            #pragma unroll
            for (int j = 0; j < 4 && (i + j) < elems_per_thread; ++j) {
                int smem_row = row;
                int smem_col = col + j;
                if (smem_row < kBlockM && smem_col < kHeadDim) {
                    // Apply swizzle pattern (Swizzle<3,3,3> for 128B)
                    int swizzled_col = smem_col ^ ((smem_row & 7) << 3);
                    smem_o[smem_row * kHeadDim + swizzled_col] = dst_arr[j];
                }
            }
        }

        cutlass::arch::fence_view_async_shared();
    }

    ///////////////////////////////////////////////////////////////////////////
    // TMA store from SMEM to GMEM
    ///////////////////////////////////////////////////////////////////////////
    template <typename WorkTileInfo, typename SchedulerParams>
    CUTLASS_DEVICE void
    tma_store(
        SharedStorage& shared_storage,
        Params const& epilogue_params,
        WorkTileInfo work_tile_info,
        SchedulerParams const& scheduler_params,
        int thread_idx
    ) {
        auto [m_block, bidh, bidb] = work_tile_info.get_block_coord(scheduler_params);
        Tensor sO = cute::as_position_independent_swizzle_tensor(
            make_tensor(make_smem_ptr(shared_storage.smem_o.data()), SmemLayoutO{}));
        Tensor mO = epilogue_params.tma_store_O.get_tma_tensor(epilogue_params.shape_O);
        Tensor gO = local_tile(mO(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));
        auto block_tma_O = epilogue_params.tma_store_O.get_slice(_0{});
        Tensor tOgO = block_tma_O.partition_D(gO);
        Tensor tOsO = block_tma_O.partition_S(sO);

        cute::copy(epilogue_params.tma_store_O, tOsO, tOgO);
        tma_store_arrive();
    }

    CUTLASS_DEVICE void store_tail() {
        tma_store_wait<0>();
    }

    // Write zeros to output (for early exit in causal masking)
    CUTLASS_DEVICE void
    store_zero(
        Params const& epilogue_params,
        int thread_idx,
        cute::tuple<int32_t, int32_t, int32_t> const& block_coord
    ) {
        auto [m_block, bidh, bidb] = block_coord;
        Tensor mO = make_tensor(make_gmem_ptr(epilogue_params.ptr_O),
                                epilogue_params.shape_O, epilogue_params.stride_O);
        Tensor gO = local_tile(mO(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));
        auto shape_LSE = select<0, 2, 3>(epilogue_params.shape_O);
        Tensor mLSE = make_tensor(make_gmem_ptr(epilogue_params.ptr_LSE), shape_LSE, epilogue_params.stride_LSE);
        Tensor gLSE = local_tile(mLSE(_, bidh, bidb), Shape<Int<kBlockM>>{}, make_coord(m_block));

        GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_fragment_like(tOgO);
        clear(tOrO);

        Tensor cO = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));

        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) {
            tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(epilogue_params.shape_O);
        }

        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, get<0>(epilogue_params.shape_O) - m_block * kBlockM
        );

        static_assert(kBlockM <= NumMmaThreads);
        if (thread_idx < get<0>(shape_LSE) - m_block * kBlockM) {
            gLSE(thread_idx) = INFINITY;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Store LSE (log-sum-exp) values
    ///////////////////////////////////////////////////////////////////////////
    template <typename WorkTileInfo, typename SchedulerParams>
    CUTLASS_DEVICE void
    store_lse(
        Params const& epilogue_params,
        WorkTileInfo work_tile_info,
        SchedulerParams const& scheduler_params,
        float const* row_sum,
        float const* row_max,
        int rows_per_thread,
        int thread_idx
    ) {
        auto [m_block, bidh, bidb] = work_tile_info.get_block_coord(scheduler_params);
        auto shape_LSE = select<0, 2, 3>(epilogue_params.shape_O);
        Tensor mLSE = make_tensor(make_gmem_ptr(epilogue_params.ptr_LSE), shape_LSE, epilogue_params.stride_LSE);
        Tensor gLSE = local_tile(mLSE(_, bidh, bidb), Shape<Int<kBlockM>>{}, make_coord(m_block));

        int const seqlen_q = get<0>(shape_LSE);
        int const row_start = m_block * kBlockM;

        // Each thread computes and stores LSE for its rows
        // LSE = max + log(sum)
        #pragma unroll
        for (int i = 0; i < rows_per_thread; ++i) {
            int row = thread_idx * rows_per_thread + i;
            if (row_start + row < seqlen_q && row < kBlockM) {
                float lse = row_max[i] + logf(row_sum[i]);
                // Handle edge case where sum is 0
                if (row_sum[i] == 0.0f || row_sum[i] != row_sum[i]) {
                    lse = -INFINITY;
                }
                gLSE(row) = lse;
            }
        }
    }
};

} // namespace flash
