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
 * SM100 (B200/B300) Mainloop for FlashAttention with tcgen05 MMA.
 *
 * KEY DIFFERENCES FROM SM120:
 * ==========================
 * 1. tcgen05.mma writes accumulators to TMEM (256KB per SM), not registers
 * 2. The MMA instruction is asynchronous - requires mbarrier coordination
 * 3. Only thread 0 issues the tcgen05.mma instruction
 * 4. Data flows: GMEM -> SMEM (TMA) -> descriptors -> tcgen05.mma -> TMEM
 * 5. Results loaded from TMEM to registers via tcgen05.ld
 */

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "../blackwell/utils.h"
#include "../blackwell/named_barrier.h"
#include "cute_extension.h"

namespace flash {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// SM100-Specific Softmax for Raw Float Arrays
// Unlike SM120's SoftmaxFused which works with cute tensors, this version
// operates on plain float arrays since SM100 accumulator is in TMEM
///////////////////////////////////////////////////////////////////////////////

template <int kBlockM, int kBlockN, int kNumThreads>
struct SoftmaxSm100 {
    static constexpr int kElemsPerThread = (kBlockM * kBlockN) / kNumThreads;
    static constexpr int kRowsPerThread = kBlockM / (kNumThreads / 4);  // 4 threads per row for reduction
    static constexpr float kLog2e = 1.4426950408889634f;
    static constexpr float kFp4Scale = 6.0f;  // Max FP4 e2m1 value
    static constexpr float kFp8Fp4ScaleLog2 = -11.392317422778762f;  // log2(1/(448*6))

    // Per-row state (each thread handles subset of rows)
    float row_max[kRowsPerThread];
    float row_sum[kRowsPerThread];
    float scores_scale[kRowsPerThread];

    CUTLASS_DEVICE SoftmaxSm100() {
        #pragma unroll
        for (int i = 0; i < kRowsPerThread; ++i) {
            row_max[i] = -INFINITY;
            row_sum[i] = 0.0f;
            scores_scale[i] = 1.0f;
        }
    }

    // Compute online softmax with output rescaling
    // acc_s: attention scores (kElemsPerThread elements per thread)
    // acc_o: output accumulator (may be different size)
    // is_first: true for first K/V block
    template <bool scale_output>
    CUTLASS_DEVICE void online_softmax(
        float* acc_s,
        int num_s,
        float* acc_o,
        int num_o,
        bool is_first,
        float softmax_scale_log2
    ) {
        int const thread_idx = threadIdx.x % kNumThreads;
        int const start_elem = thread_idx * kElemsPerThread;

        if (is_first) {
            // First block: compute max, subtract, exp, sum
            #pragma unroll
            for (int i = 0; i < num_s; ++i) {
                int elem_idx = start_elem + i;
                int row = elem_idx / kBlockN;
                int local_row = row % kRowsPerThread;
                row_max[local_row] = fmaxf(row_max[local_row], acc_s[i]);
            }

            // Warp shuffle to get max across threads handling same row
            #pragma unroll
            for (int i = 0; i < kRowsPerThread; ++i) {
                row_max[i] = fmaxf(row_max[i], __shfl_xor_sync(0xffffffff, row_max[i], 1));
                row_max[i] = fmaxf(row_max[i], __shfl_xor_sync(0xffffffff, row_max[i], 2));
            }

            // Apply exp2 and sum
            #pragma unroll
            for (int i = 0; i < num_s; ++i) {
                int elem_idx = start_elem + i;
                int row = elem_idx / kBlockN;
                int local_row = row % kRowsPerThread;
                float max_scaled = (row_max[local_row] == -INFINITY) ? 0.0f :
                                   row_max[local_row] * softmax_scale_log2;
                acc_s[i] = exp2f(acc_s[i] * softmax_scale_log2 - max_scaled);
                row_sum[local_row] += acc_s[i];
            }
        } else {
            // Subsequent blocks: track new max, rescale previous results
            float prev_max[kRowsPerThread];
            #pragma unroll
            for (int i = 0; i < kRowsPerThread; ++i) {
                prev_max[i] = row_max[i];
            }

            // Find new max
            #pragma unroll
            for (int i = 0; i < num_s; ++i) {
                int elem_idx = start_elem + i;
                int row = elem_idx / kBlockN;
                int local_row = row % kRowsPerThread;
                row_max[local_row] = fmaxf(row_max[local_row], acc_s[i]);
            }

            // Warp shuffle for max
            #pragma unroll
            for (int i = 0; i < kRowsPerThread; ++i) {
                row_max[i] = fmaxf(row_max[i], __shfl_xor_sync(0xffffffff, row_max[i], 1));
                row_max[i] = fmaxf(row_max[i], __shfl_xor_sync(0xffffffff, row_max[i], 2));
            }

            // Compute scale factor for previous results
            #pragma unroll
            for (int i = 0; i < kRowsPerThread; ++i) {
                float new_max = (row_max[i] == -INFINITY) ? 0.0f : row_max[i];
                float old_max = (prev_max[i] == -INFINITY) ? 0.0f : prev_max[i];
                scores_scale[i] = exp2f((old_max - new_max) * softmax_scale_log2);
                row_sum[i] *= scores_scale[i];
            }

            // Rescale output accumulator
            if constexpr (scale_output) {
                int o_start = thread_idx * (num_o / kNumThreads);
                #pragma unroll
                for (int i = 0; i < num_o; ++i) {
                    int elem_idx = o_start + i;
                    int row = elem_idx / (num_o / kBlockM * kNumThreads / kBlockM);
                    int local_row = row % kRowsPerThread;
                    acc_o[i] *= scores_scale[local_row];
                }
            }

            // Apply exp2 and sum for new scores
            #pragma unroll
            for (int i = 0; i < num_s; ++i) {
                int elem_idx = start_elem + i;
                int row = elem_idx / kBlockN;
                int local_row = row % kRowsPerThread;
                float max_scaled = (row_max[local_row] == -INFINITY) ? 0.0f :
                                   row_max[local_row] * softmax_scale_log2;
                acc_s[i] = exp2f(acc_s[i] * softmax_scale_log2 - max_scaled);
                row_sum[local_row] += acc_s[i];
            }
        }
    }

    // Finalize: divide output by sum
    CUTLASS_DEVICE void finalize(float* acc_o, int num_o) {
        int const thread_idx = threadIdx.x % kNumThreads;

        // Reduce sum across threads
        #pragma unroll
        for (int i = 0; i < kRowsPerThread; ++i) {
            row_sum[i] += __shfl_xor_sync(0xffffffff, row_sum[i], 1);
            row_sum[i] += __shfl_xor_sync(0xffffffff, row_sum[i], 2);
        }

        // Divide by sum
        int o_start = thread_idx * (num_o / kNumThreads);
        #pragma unroll
        for (int i = 0; i < num_o; ++i) {
            int elem_idx = o_start + i;
            int row = elem_idx / (num_o / kBlockM);
            int local_row = row % kRowsPerThread;
            float inv_sum = (row_sum[local_row] == 0.0f || row_sum[local_row] != row_sum[local_row])
                           ? 0.0f : 1.0f / row_sum[local_row];
            acc_o[i] *= inv_sum;
        }
    }

    // Get scale factor for rescaling output after P*V
    CUTLASS_DEVICE float get_scale(int row) {
        return scores_scale[row % kRowsPerThread];
    }
};

///////////////////////////////////////////////////////////////////////////////
// SM100 Collective Mainloop for Flash Attention Forward Pass
// Uses tcgen05.mma with TMEM for accumulator storage
///////////////////////////////////////////////////////////////////////////////

template <typename Ktraits, bool Is_causal>
struct CollectiveMainloopFwdSm100 {

    using Element = typename Ktraits::Element;
    using ElementSF = typename Ktraits::ElementSF;
    using ElementAccum = typename Ktraits::ElementAccum;
    using ElementOut = typename Ktraits::ElementOut;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static constexpr int kStages = Ktraits::kStages;
    static constexpr int kHeadDim = Ktraits::kHeadDim;
    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockN = Ktraits::kBlockN;
    static constexpr bool per_block_mean = Ktraits::per_block_mean;
    static constexpr int kSFVecSize = Ktraits::kSFVecSize;

    // Thread configuration
    static constexpr int kNThreads = Ktraits::kNThreads;
    static constexpr int NumMmaThreads = kNThreads - cutlass::NumThreadsPerWarpGroup;

    // Instruction descriptors
    static constexpr uint32_t kInstrDescQK = Ktraits::kInstrDescQK;
    static constexpr uint32_t kInstrDescPV = Ktraits::kInstrDescPV;

    // TMA copy operations
    using GmemTiledCopy = typename Ktraits::GmemTiledCopyQ;

    // Shared memory layouts
    using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
    using SmemLayoutK = typename Ktraits::SmemLayoutK;
    using SmemLayoutV = typename Ktraits::SmemLayoutV;
    using SmemLayoutP = typename Ktraits::SmemLayoutP;
    using SmemLayoutO = typename Ktraits::SmemLayoutO;

    // Scale factor layouts
    using SmemLayoutSFQ = typename Ktraits::SmemLayoutSFQ;
    using SmemLayoutSFK = typename Ktraits::SmemLayoutSFK;
    using SmemLayoutSFV = typename Ktraits::SmemLayoutSFV;
    using SmemLayoutSFP = typename Ktraits::SmemLayoutSFP;

    // Delta_S layout
    using SmemLayoutDS = typename Ktraits::SmemLayoutDS;

    // Shape and stride types
    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapeSF = cute::Shape<int32_t, int32_t, int32_t, int32_t>;

    // TMA descriptors
    using TMA_Q = decltype(make_tma_copy(
        GmemTiledCopy{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                    repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{}),
        SmemLayoutQ{},
        select<0, 2>(TileShape_MNK{}),
        _1{}));

    using TMA_K = decltype(make_tma_copy(
        GmemTiledCopy{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                    repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{}),
        take<0, 2>(SmemLayoutK{}),
        select<1, 2>(TileShape_MNK{}),
        _1{}));

    using TMA_V = decltype(make_tma_copy(
        GmemTiledCopy{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                    repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{}),
        take<0, 2>(SmemLayoutV{}),
        make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
        _1{}));

    // TMA transaction sizes
    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(
        cutlass::bits_to_bytes(size(SmemLayoutQ{}) * sizeof_bits<Element>::value));

    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(
        cutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutK{})) * sizeof_bits<Element>::value));

    static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(
        cutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutV{})) * sizeof_bits<Element>::value));

    // Pipeline types
    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using MainloopPipelineQ = typename Ktraits::MainloopPipelineQ;
    using PipelineStateQ = typename Ktraits::PipelineStateQ;

    // Softmax type
    using Softmax = SoftmaxSm100<kBlockM, kBlockN, NumMmaThreads>;

    ///////////////////////////////////////////////////////////////////////////
    // Host-side kernel arguments
    ///////////////////////////////////////////////////////////////////////////
    struct Arguments {
        Element const* ptr_Q;
        ShapeQKV const shape_Q;
        StrideQKV const stride_Q;
        Element const* ptr_K;
        ShapeQKV const shape_K;
        StrideQKV const stride_K;
        ShapeQKV const unpadded_shape_K;
        Element const* ptr_V;
        ShapeQKV const shape_V;
        StrideQKV const stride_V;
        ElementSF const* ptr_SFQ;
        ShapeSF const shape_SFQ;
        ElementSF const* ptr_SFK;
        ShapeSF const shape_SFK;
        ElementSF const* ptr_SFV;
        ShapeSF const shape_SFV;
        float const* ptr_delta_s;
        ShapeQKV const shape_delta_s;
        StrideQKV const stride_delta_s;
        float const softmax_scale_log2;
    };

    ///////////////////////////////////////////////////////////////////////////
    // Device-side kernel parameters
    ///////////////////////////////////////////////////////////////////////////
    struct Params {
        ShapeQKV const shape_Q;
        ShapeQKV const shape_K;
        ShapeQKV const unpadded_shape_K;
        ShapeQKV const shape_V;
        TMA_Q tma_load_Q;
        TMA_K tma_load_K;
        TMA_V tma_load_V;
        ElementSF const* ptr_SFQ;
        ElementSF const* ptr_SFK;
        ElementSF const* ptr_SFV;
        float const* ptr_delta_s;
        StrideQKV const stride_delta_s;
        float const softmax_scale_log2;
    };

    ///////////////////////////////////////////////////////////////////////////
    // Convert Arguments to Params (host-side)
    ///////////////////////////////////////////////////////////////////////////
    static Params to_underlying_arguments(Arguments const& args) {
        Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
        TMA_Q tma_load_Q = make_tma_copy(
            GmemTiledCopy{}, mQ, SmemLayoutQ{},
            select<0, 2>(TileShape_MNK{}), _1{});

        Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
        TMA_K tma_load_K = make_tma_copy(
            GmemTiledCopy{}, mK, take<0, 2>(SmemLayoutK{}),
            select<1, 2>(TileShape_MNK{}), _1{});

        Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V), args.shape_V, args.stride_V);
        TMA_V tma_load_V = make_tma_copy(
            GmemTiledCopy{}, mV, take<0, 2>(SmemLayoutV{}),
            make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})), _1{});

        return {
            args.shape_Q,
            args.shape_K,
            args.unpadded_shape_K,
            args.shape_V,
            tma_load_Q,
            tma_load_K,
            tma_load_V,
            args.ptr_SFQ,
            args.ptr_SFK,
            args.ptr_SFV,
            args.ptr_delta_s,
            args.stride_delta_s,
            args.softmax_scale_log2
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // Prefetch TMA descriptors
    ///////////////////////////////////////////////////////////////////////////
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
    }

    ///////////////////////////////////////////////////////////////////////////
    // Get maximum N block for causal masking
    ///////////////////////////////////////////////////////////////////////////
    CUTLASS_DEVICE
    int get_n_block_max(Params const& params, int m_block) const {
        int const seqlen_q = get<0>(params.shape_Q);
        int const seqlen_k = get<0>(params.shape_K);
        int n_block_max = cute::ceil_div(seqlen_k, kBlockN);
        if constexpr (Is_causal) {
            n_block_max = std::min(n_block_max,
                cute::ceil_div((m_block + 1) * kBlockM + seqlen_k - seqlen_q, kBlockN));
        }
        return n_block_max;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Initialize TMEM and mbarriers (called once per tile by consumer thread 0)
    ///////////////////////////////////////////////////////////////////////////
    template <typename SharedStorage>
    CUTLASS_DEVICE
    static void init_tmem_and_mbarriers(SharedStorage& shared_storage, int thread_idx) {
        // Only consumer thread 0 initializes
        if (thread_idx != 0) return;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
        // Allocate TMEM for accumulators
        cute::TmemAllocatorSm100 alloc_qk, alloc_pv;

        // QK accumulator: kBlockM × kBlockN floats
        uint32_t cols_qk = Ktraits::kTmemColsAccQK;
        shared_storage.tmem_acc_qk = alloc_qk.allocate(cols_qk);

        // PV accumulator: kBlockM × kHeadDim floats
        uint32_t cols_pv = Ktraits::kTmemColsAccPV;
        shared_storage.tmem_acc_pv = alloc_pv.allocate(cols_pv);

        // Scale factor TMEM (smaller allocations)
        cute::TmemAllocatorSm100 alloc_sf;
        shared_storage.tmem_sf_q = alloc_sf.allocate(Ktraits::kTmemColsSF);
        shared_storage.tmem_sf_k = alloc_sf.allocate(Ktraits::kTmemColsSF);
        shared_storage.tmem_sf_v = alloc_sf.allocate(Ktraits::kTmemColsSF);
        shared_storage.tmem_sf_p = alloc_sf.allocate(Ktraits::kTmemColsSF);

        // Initialize mbarriers for MMA synchronization
        uint32_t mbar_qk_0 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&shared_storage.mbar_qk[0]));
        uint32_t mbar_qk_1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&shared_storage.mbar_qk[1]));
        uint32_t mbar_pv_0 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&shared_storage.mbar_pv[0]));
        uint32_t mbar_pv_1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&shared_storage.mbar_pv[1]));

        cute::mbarrier_init(mbar_qk_0, 1);
        cute::mbarrier_init(mbar_qk_1, 1);
        cute::mbarrier_init(mbar_pv_0, 1);
        cute::mbarrier_init(mbar_pv_1, 1);

        shared_storage.mma_phase_qk = 0;
        shared_storage.mma_phase_pv = 0;
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    // Deallocate TMEM (called at end of tile processing)
    ///////////////////////////////////////////////////////////////////////////
    template <typename SharedStorage>
    CUTLASS_DEVICE
    static void cleanup_tmem(SharedStorage& shared_storage, int thread_idx) {
        if (thread_idx != 0) return;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
        // Deallocate TMEM
        cute::TmemAllocatorSm100 alloc;
        alloc.base_addr_ = shared_storage.tmem_acc_qk;
        alloc.num_cols_ = Ktraits::kTmemColsAccQK;
        alloc.deallocate();

        alloc.base_addr_ = shared_storage.tmem_acc_pv;
        alloc.num_cols_ = Ktraits::kTmemColsAccPV;
        alloc.deallocate();
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    // Producer: Load Q, K, V tiles from global memory to shared memory
    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulerParams, typename SharedStorage, typename WorkTileInfo>
    CUTLASS_DEVICE void
    load(Params const& params,
         SchedulerParams const& scheduler_params,
         MainloopPipelineQ pipeline_q,
         MainloopPipeline pipeline_k,
         MainloopPipeline pipeline_v,
         PipelineStateQ& smem_pipe_write_q,
         PipelineState& smem_pipe_write_k,
         PipelineState& smem_pipe_write_v,
         SharedStorage& shared_storage,
         WorkTileInfo work_tile_info,
         int& work_idx,
         int& tile_count_semaphore) {

        auto [m_block, bidh, bidb] = work_tile_info.get_block_coord(scheduler_params);
        int n_block_max = get_n_block_max(params, m_block);

        // Create SMEM tensors
        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

        // Get TMA tensors
        Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q);
        Tensor mK = params.tma_load_K.get_tma_tensor(params.shape_K);
        Tensor mV = params.tma_load_V.get_tma_tensor(params.shape_V);

        // Partition for this tile
        Tensor gQ = local_tile(mQ(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}),
                               make_coord(m_block, _0{}));
        Tensor gK = local_tile(mK(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}),
                               make_coord(_, _0{}));
        Tensor gV = local_tile(mV(_, _, bidh, bidb),
                               make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
                               make_coord(_0{}, _));

        // TMA partition
        auto block_tma_q = params.tma_load_Q.get_slice(_0{});
        Tensor tQgQ = block_tma_q.partition_S(gQ);
        Tensor tQsQ = block_tma_q.partition_D(sQ);

        auto block_tma_k = params.tma_load_K.get_slice(_0{});
        Tensor tKgK = group_modes<0, 3>(block_tma_k.partition_S(gK));
        Tensor tKsK = group_modes<0, 3>(block_tma_k.partition_D(sK));

        auto block_tma_v = params.tma_load_V.get_slice(_0{});
        Tensor tVgV = group_modes<0, 3>(block_tma_v.partition_S(gV));
        Tensor tVsV = group_modes<0, 3>(block_tma_v.partition_D(sV));

        int n_block = n_block_max - 1;
        int lane_predicate = cute::elect_one_sync();

        if (lane_predicate) {
            // Load Q (single stage)
            pipeline_q.producer_acquire(smem_pipe_write_q);
            copy(params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q), 0),
                 tQgQ, tQsQ);
            ++smem_pipe_write_q;

            // Load first K tile
            pipeline_k.producer_acquire(smem_pipe_write_k);
            copy(params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), 0),
                 tKgK(_, n_block), tKsK(_, smem_pipe_write_k.index()));
            ++smem_pipe_write_k;

            // Load first V tile
            pipeline_v.producer_acquire(smem_pipe_write_v);
            copy(params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), 0),
                 tVgV(_, n_block), tVsV(_, smem_pipe_write_v.index()));
            ++smem_pipe_write_v;
        }

        --n_block;

        // Load remaining K/V tiles
        if (lane_predicate) {
            #pragma unroll 2
            for (; n_block >= 0; --n_block) {
                pipeline_k.producer_acquire(smem_pipe_write_k);
                copy(params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), 0),
                     tKgK(_, n_block), tKsK(_, smem_pipe_write_k.index()));
                ++smem_pipe_write_k;

                pipeline_v.producer_acquire(smem_pipe_write_v);
                copy(params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), 0),
                     tVgV(_, n_block), tVsV(_, smem_pipe_write_v.index()));
                ++smem_pipe_write_v;
            }
        }
        ++work_idx;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Producer tail
    ///////////////////////////////////////////////////////////////////////////
    CUTLASS_DEVICE void
    load_tail(MainloopPipelineQ pipeline_q,
              MainloopPipeline pipeline_k,
              MainloopPipeline pipeline_v,
              PipelineStateQ& smem_pipe_write_q,
              PipelineState& smem_pipe_write_k,
              PipelineState& smem_pipe_write_v) {
        int lane_predicate = cute::elect_one_sync();
        if (lane_predicate) {
            pipeline_q.producer_tail(smem_pipe_write_q);
            pipeline_k.producer_tail(smem_pipe_write_k);
            pipeline_v.producer_tail(smem_pipe_write_v);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Issue tcgen05.mma for Q*K^T (called by thread 0 only)
    ///////////////////////////////////////////////////////////////////////////
    template <typename SharedStorage>
    CUTLASS_DEVICE
    static void issue_mma_qk(SharedStorage& shared_storage, int stage_k, bool accumulate) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
        // Get SMEM pointers
        void const* q_ptr = shared_storage.smem_q.data();
        void const* k_ptr = shared_storage.smem_k.data() +
                            stage_k * cosize(take<0,2>(SmemLayoutK{}));
        void const* sfq_ptr = shared_storage.smem_sfq.data();
        void const* sfk_ptr = shared_storage.smem_sfk.data() +
                              stage_k * kBlockN * (kHeadDim / kSFVecSize);

        // Create SMEM descriptors
        uint64_t q_desc = cute::make_q_smem_desc(q_ptr, kBlockM, kHeadDim);
        uint64_t k_desc = cute::make_k_smem_desc(k_ptr, kBlockN, kHeadDim);

        // Copy scale factors to TMEM
        cute::copy_sf_smem_to_tmem(shared_storage.tmem_sf_q, sfq_ptr,
                             kBlockM, kHeadDim / kSFVecSize);
        cute::copy_sf_smem_to_tmem(shared_storage.tmem_sf_k, sfk_ptr,
                             kBlockN, kHeadDim / kSFVecSize);

        // Issue MMA
        cute::Sm100BlockscaledMma::mma(
            shared_storage.tmem_acc_qk,
            q_desc,
            k_desc,
            kInstrDescQK,
            shared_storage.tmem_sf_q,
            shared_storage.tmem_sf_k,
            accumulate
        );

        // Signal completion
        uint32_t mbar_addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&shared_storage.mbar_qk[shared_storage.mma_phase_qk & 1]));
        cute::umma_commit(mbar_addr);
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    // Issue tcgen05.mma for P*V (called by thread 0 only)
    ///////////////////////////////////////////////////////////////////////////
    template <typename SharedStorage>
    CUTLASS_DEVICE
    static void issue_mma_pv(SharedStorage& shared_storage, int stage_v, bool accumulate) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
        // Get SMEM pointers - P is in union with Q
        void const* p_ptr = shared_storage.smem_p.data();
        void const* v_ptr = shared_storage.smem_v.data() +
                            stage_v * cosize(take<0,2>(SmemLayoutV{}));
        void const* sfp_ptr = shared_storage.smem_sfp.data();
        void const* sfv_ptr = shared_storage.smem_sfv.data() +
                              stage_v * kBlockN * (kHeadDim / kSFVecSize);

        // Create SMEM descriptors
        uint64_t p_desc = cute::make_p_smem_desc(p_ptr, kBlockM, kBlockN);
        uint64_t v_desc = cute::make_v_smem_desc(v_ptr, kHeadDim, kBlockN);

        // Copy scale factors to TMEM
        cute::copy_sf_smem_to_tmem(shared_storage.tmem_sf_p, sfp_ptr,
                             kBlockM, kBlockN / kSFVecSize);
        cute::copy_sf_smem_to_tmem(shared_storage.tmem_sf_v, sfv_ptr,
                             kBlockN, kHeadDim / kSFVecSize);

        // Issue MMA
        cute::Sm100BlockscaledMma::mma(
            shared_storage.tmem_acc_pv,
            p_desc,
            v_desc,
            kInstrDescPV,
            shared_storage.tmem_sf_p,
            shared_storage.tmem_sf_v,
            accumulate
        );

        // Signal completion
        uint32_t mbar_addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&shared_storage.mbar_pv[shared_storage.mma_phase_pv & 1]));
        cute::umma_commit(mbar_addr);
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    // Wait for MMA completion and load results from TMEM
    ///////////////////////////////////////////////////////////////////////////
    template <typename SharedStorage>
    CUTLASS_DEVICE
    static void wait_mma_qk(SharedStorage& shared_storage, float* acc_data, int thread_idx) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
        // Wait on mbarrier
        uint32_t mbar_addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&shared_storage.mbar_qk[shared_storage.mma_phase_qk & 1]));
        cute::mbarrier_wait(mbar_addr, shared_storage.mma_phase_qk & 1);

        // Load from TMEM - each thread loads its portion
        // TMEM layout: 128 rows × N columns
        int const elems_per_thread = (kBlockM * kBlockN) / NumMmaThreads;

        uint32_t tmem_base = shared_storage.tmem_acc_qk;
        int start_elem = thread_idx * elems_per_thread;

        #pragma unroll
        for (int i = 0; i < elems_per_thread; i += 8) {
            int elem_idx = start_elem + i;
            int row = elem_idx / kBlockN;
            int col = elem_idx % kBlockN;
            uint32_t addr = cute::TmemAllocatorSm100::addr_at(tmem_base, row % 128, col);
            cute::tmem_load_8xf32(&acc_data[i], addr);
        }
        cute::tmem_load_wait();

        // Toggle phase
        if (thread_idx == 0) {
            shared_storage.mma_phase_qk++;
        }
#endif
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE
    static void wait_mma_pv(SharedStorage& shared_storage, float* acc_data, int thread_idx) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
        uint32_t mbar_addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&shared_storage.mbar_pv[shared_storage.mma_phase_pv & 1]));
        cute::mbarrier_wait(mbar_addr, shared_storage.mma_phase_pv & 1);

        int const elems_per_thread = (kBlockM * kHeadDim) / NumMmaThreads;

        uint32_t tmem_base = shared_storage.tmem_acc_pv;
        int start_elem = thread_idx * elems_per_thread;

        #pragma unroll
        for (int i = 0; i < elems_per_thread; i += 8) {
            int elem_idx = start_elem + i;
            int row = elem_idx / kHeadDim;
            int col = elem_idx % kHeadDim;
            uint32_t addr = cute::TmemAllocatorSm100::addr_at(tmem_base, row % 128, col);
            cute::tmem_load_8xf32(&acc_data[i], addr);
        }
        cute::tmem_load_wait();

        if (thread_idx == 0) {
            shared_storage.mma_phase_pv++;
        }
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    // Quantize P matrix to FP4 and compute scale factors
    // Uses packed_float_to_e2m1 and packed_float_to_ue4m3 from utils.h
    ///////////////////////////////////////////////////////////////////////////
    template <typename SharedStorage>
    CUTLASS_DEVICE
    static void quantize_p_to_smem(
        SharedStorage& shared_storage,
        float const* acc_p,
        int num_elems,
        int thread_idx
    ) {
        // Get SMEM pointers for P and its scale factors
        Element* p_smem = shared_storage.smem_p.data();
        ElementSF* sfp_smem = shared_storage.smem_sfp.data();

        int const elems_per_thread = num_elems;
        int const start_elem = thread_idx * elems_per_thread;

        // Process in groups of 16 (one scale factor per 16 elements)
        #pragma unroll
        for (int sf_idx = 0; sf_idx < elems_per_thread / kSFVecSize; ++sf_idx) {
            int base = sf_idx * kSFVecSize;

            // Find max absolute value for scale factor
            float max_abs = 0.0f;
            #pragma unroll
            for (int j = 0; j < kSFVecSize; ++j) {
                max_abs = fmaxf(max_abs, fabsf(acc_p[base + j]));
            }

            // Compute scale factor
            // FP4 e2m1 range is [-6, 6], so scale = max_abs / 6.0
            float scale = max_abs / 6.0f;
            scale = fmaxf(scale, 1e-12f);  // Avoid division by zero
            float inv_scale = 1.0f / scale;

            // Convert scale to FP8 E4M3 and store
            // The scale factor needs to be stored such that: quantized_value * scale = original
            uint32_t sf_packed;
            packed_float_to_ue4m3(scale, scale, scale, scale, sf_packed);
            int sf_smem_idx = (start_elem + base) / kSFVecSize;
            if (sf_smem_idx < kBlockM * kBlockN / kSFVecSize) {
                // Store just the first byte (one FP8 value)
                sfp_smem[sf_smem_idx] = *reinterpret_cast<ElementSF*>(&sf_packed);
            }

            // Quantize 16 values to FP4 (8 packed bytes)
            // packed_float_to_e2m1 takes 8 floats and produces 4 bytes (8 FP4 values)
            #pragma unroll
            for (int k = 0; k < kSFVecSize; k += 8) {
                // Scale values to FP4 range
                float f0 = acc_p[base + k + 0] * inv_scale;
                float f1 = acc_p[base + k + 1] * inv_scale;
                float f2 = acc_p[base + k + 2] * inv_scale;
                float f3 = acc_p[base + k + 3] * inv_scale;
                float f4 = acc_p[base + k + 4] * inv_scale;
                float f5 = acc_p[base + k + 5] * inv_scale;
                float f6 = acc_p[base + k + 6] * inv_scale;
                float f7 = acc_p[base + k + 7] * inv_scale;

                // Convert to FP4 e2m1 packed format
                uint32_t packed;
                packed_float_to_e2m1(f0, f1, f2, f3, f4, f5, f6, f7, packed);

                // Store 4 bytes (8 FP4 values)
                int p_smem_idx = (start_elem + base + k) / 2;  // 2 values per byte
                if (p_smem_idx + 3 < kBlockM * kBlockN / 2) {
                    *reinterpret_cast<uint32_t*>(&p_smem[p_smem_idx]) = packed;
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Consumer: Main attention computation loop
    ///////////////////////////////////////////////////////////////////////////
    template <typename SharedStorage, typename FrgTensorO>
    CUTLASS_DEVICE void
    mma(Params const& params,
        MainloopPipelineQ pipeline_q,
        MainloopPipeline pipeline_k,
        MainloopPipeline pipeline_v,
        PipelineStateQ& smem_pipe_read_q,
        PipelineState& smem_pipe_read_k,
        PipelineState& smem_pipe_read_v,
        FrgTensorO& tOrO,
        Softmax& softmax,
        int n_block_count,
        int thread_idx,
        int work_idx,
        int m_block,
        SharedStorage& shared_storage) {

        int const seqlen_q = get<0>(params.shape_Q);
        int const seqlen_k = get<0>(params.shape_K);
        int const unpadded_seqlen_k = get<0>(params.unpadded_shape_K);
        int n_block = n_block_count - 1;

        // Consumer thread index (relative to consumer warp groups)
        int const consumer_thread_idx = thread_idx;
        bool const is_thread_0 = (consumer_thread_idx == 0);

        // Local accumulator storage
        int const elems_per_thread_qk = (kBlockM * kBlockN) / NumMmaThreads;
        int const elems_per_thread_pv = (kBlockM * kHeadDim) / NumMmaThreads;
        float acc_qk[elems_per_thread_qk];
        float acc_pv[elems_per_thread_pv];

        // Initialize PV accumulator to zero
        #pragma unroll
        for (int i = 0; i < elems_per_thread_pv; ++i) {
            acc_pv[i] = 0.0f;
        }

        // Helper for pipeline wait
        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        // Wait for Q to arrive
        consumer_wait(pipeline_q, smem_pipe_read_q);
        pipeline_q.consumer_release(smem_pipe_read_q);
        ++smem_pipe_read_q;

        // Causal masking helper
        auto col_limit_causal = [&](int row, int n_blk) {
            return row + 1 + seqlen_k - n_blk * kBlockN - seqlen_q + m_block * kBlockM;
        };

        //=====================================================================
        // Main loop over K/V blocks
        //=====================================================================
        for (; n_block >= 0; --n_block) {
            // Wait for K tile
            consumer_wait(pipeline_k, smem_pipe_read_k);

            //------------------------------------------------------------------
            // GEMM-I: Q * K^T -> S (attention scores)
            //------------------------------------------------------------------
            __syncthreads();

            if (is_thread_0) {
                issue_mma_qk(shared_storage, smem_pipe_read_k.index(), false);
            }

            // Wait for MMA completion and load from TMEM
            wait_mma_qk(shared_storage, acc_qk, consumer_thread_idx);

            pipeline_k.consumer_release(smem_pipe_read_k);
            ++smem_pipe_read_k;

            //------------------------------------------------------------------
            // Add delta_s (per-block mean) if enabled
            //------------------------------------------------------------------
            if constexpr (per_block_mean) {
                // Load delta_s from params and add to acc_qk
                float const* delta_s_ptr = params.ptr_delta_s +
                    m_block * get<0>(params.stride_delta_s) +
                    n_block * kBlockN;

                int start_elem = consumer_thread_idx * elems_per_thread_qk;
                #pragma unroll
                for (int i = 0; i < elems_per_thread_qk; ++i) {
                    int elem_idx = start_elem + i;
                    int col = elem_idx % kBlockN;
                    acc_qk[i] += delta_s_ptr[col];
                }
            }

            //------------------------------------------------------------------
            // Apply causal masking
            //------------------------------------------------------------------
            {
                int start_elem = consumer_thread_idx * elems_per_thread_qk;
                #pragma unroll
                for (int i = 0; i < elems_per_thread_qk; ++i) {
                    int elem_idx = start_elem + i;
                    int row = elem_idx / kBlockN;
                    int col = elem_idx % kBlockN;

                    // Apply masking
                    if constexpr (!Is_causal) {
                        if (col >= unpadded_seqlen_k - n_block * kBlockN) {
                            acc_qk[i] = -INFINITY;
                        }
                    } else {
                        int limit = col_limit_causal(row, n_block);
                        if (col >= std::min(seqlen_k - n_block * kBlockN, limit)) {
                            acc_qk[i] = -INFINITY;
                        }
                    }
                }
            }

            //------------------------------------------------------------------
            // Softmax: compute P = softmax(S) with online rescaling
            //------------------------------------------------------------------
            bool is_first_block = (n_block == n_block_count - 1);
            softmax.template online_softmax</*scale_output=*/true>(
                acc_qk, elems_per_thread_qk,
                acc_pv, elems_per_thread_pv,
                is_first_block,
                params.softmax_scale_log2
            );

            //------------------------------------------------------------------
            // Quantize P to FP4 and write to SMEM
            //------------------------------------------------------------------
            quantize_p_to_smem(shared_storage, acc_qk, elems_per_thread_qk, consumer_thread_idx);
            __syncthreads();

            //------------------------------------------------------------------
            // Wait for V tile
            //------------------------------------------------------------------
            consumer_wait(pipeline_v, smem_pipe_read_v);

            //------------------------------------------------------------------
            // GEMM-II: P * V -> O (accumulate output)
            //------------------------------------------------------------------
            if (is_thread_0) {
                issue_mma_pv(shared_storage, smem_pipe_read_v.index(), n_block < n_block_count - 1);
            }

            // Wait and load PV result
            float pv_result[elems_per_thread_pv];
            wait_mma_pv(shared_storage, pv_result, consumer_thread_idx);

            // Accumulate into output (with rescaling for online softmax)
            if (n_block < n_block_count - 1) {
                // Previous accumulator was already rescaled in softmax
                #pragma unroll
                for (int i = 0; i < elems_per_thread_pv; ++i) {
                    acc_pv[i] += pv_result[i];
                }
            } else {
                // First block, just copy
                #pragma unroll
                for (int i = 0; i < elems_per_thread_pv; ++i) {
                    acc_pv[i] = pv_result[i];
                }
            }

            pipeline_v.consumer_release(smem_pipe_read_v);
            ++smem_pipe_read_v;
        }

        //=====================================================================
        // Finalize output
        //=====================================================================
        // Apply final softmax normalization (divide by sum)
        softmax.finalize(acc_pv, elems_per_thread_pv);

        // Copy to output tensor
        // tOrO is a cute tensor fragment, copy our results to it
        #pragma unroll
        for (int i = 0; i < elems_per_thread_pv && i < size(tOrO); ++i) {
            tOrO(i) = acc_pv[i];
        }
    }
};

} // namespace flash
