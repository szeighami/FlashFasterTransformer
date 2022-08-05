#include "src/fastertransformer/layers/attention_layers/DecoderSelfAttentionLayer.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention.h"
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h"
//#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include <sys/time.h> // for clock_gettime()

#include <assert.h>
#include <float.h>
#include <type_traits>
#include <sstream>
#include <fstream>

#define tpb64
#define tpb128
#define tpb256
#define tpk1
#define tpv4

float half_to_float(uint16_t float16_value)
{
  // MSB -> LSB
  // float16=1bit: sign, 5bit: exponent, 10bit: fraction
  // float32=1bit: sign, 8bit: exponent, 23bit: fraction
  // for normal exponent(1 to 0x1e): value=2**(exponent-15)*(1.fraction)
  // for denormalized exponent(0): value=2**-14*(0.fraction)
  uint32_t sign = float16_value >> 15;
  uint32_t exponent = (float16_value >> 10) & 0x1F;
  uint32_t fraction = (float16_value & 0x3FF);
  uint32_t float32_value;
  if (exponent == 0)
  {
    if (fraction == 0)
    {
      // zero
      float32_value = (sign << 31);
    }
    else
    {
      // can be represented as ordinary value in float32
      // 2 ** -14 * 0.0101
      // => 2 ** -16 * 1.0100
      // int int_exponent = -14;
      exponent = 127 - 14;
      while ((fraction & (1 << 10)) == 0)
      {
        //int_exponent--;
        exponent--;
        fraction <<= 1;
      }
      fraction &= 0x3FF;
      // int_exponent += 127;
      float32_value = (sign << 31) | (exponent << 23) | (fraction << 13);  
    }    
  }
  else if (exponent == 0x1F)
  {
    /* Inf or NaN */
    float32_value = (sign << 31) | (0xFF << 23) | (fraction << 13);
  }
  else
  {
    /* ordinary number */
    float32_value = (sign << 31) | ((exponent + (127-15)) << 23) | (fraction << 13);
  }
  
  return *((float*)&float32_value);
}
using namespace fastertransformer;

//#define MMHA_USE_FP32_ACUM_FOR_OUT

namespace mmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

//
// We use the following terminology to describe the different dimensions.
//
// B:  Batch size (number of sequences),
// L:  Sequence length,
// D:  Hidden dimension,
// H:  Number of heads,
// Dh: Hidden dimension per head - Dh = D / H.
//
// The different kernels assign a threadblock for B x H pair. The grid has size (1, B, H). We use
// 64, 128 and 256 threads per block.
//
// Each threadblock loads Dh values from Q and its associated bias. The kernels run a loop to
// compute Q * K^T where K is loaded from a cache buffer -- except for the current timestep. The
// cache buffer helps with memory accesses and contains keys with bias.
//
// The layout of the cache buffer for the keys is [B, H, Dh/x, L, x] where x == 8 for FP16 and
// x == 4 for FP32 where the fastest moving dimension (contiguous data) is the rightmost one. The
// values for x are chosen to create chunks of 16 bytes.
//
// The different kernels use 1, 2 or 4 threads per key (THREADS_PER_KEY). The size of the LDGs
// depends on the number of threads per key. Each thread sums Dh / THREADS_PER_KEY elements. At
// the end of each iteration of the Q * K^T loop, we perform a reduction between lanes using an
// HMMA instruction (Tensor Core). Each Q * K^T valuey is stored in shared memory in FP32.
//
// After that loop, a parallel softmax is computed accross the different Q * K^T values stored in
// shared memory.
//
// The kernel ends with a loop over the values in V. We use THREADS_PER_VALUE to control how many
// timesteps are computed by loop iteration. As with the keys, the values are read from a cache
// except for the current timestep. The layout of the cache buffer for the values is much simpler
// as it is [B, H, L, Dh].
//

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Dh>
struct Qk_vec_ {};

template<>
struct Qk_vec_<float, 32> {
    using Type = float;
};
template<>
struct Qk_vec_<float, 64> {
    using Type = float2;
};
template<>
struct Qk_vec_<float, 128> {
    using Type = float4;
};
template<>
struct Qk_vec_<float, 256> {
    using Type = float4;
};
template<>
struct Qk_vec_<uint16_t, 32> {
    using Type = uint32_t;
};
template<>
struct Qk_vec_<uint16_t, 64> {
    using Type = uint32_t;
};
template<>
struct Qk_vec_<uint16_t, 128> {
    using Type = uint2;
};
template<>
struct Qk_vec_<uint16_t, 256> {
    using Type = uint4;
};
#ifdef ENABLE_BF16
template<>
struct Qk_vec_<__nv_bfloat16, 32> {
    using Type = __nv_bfloat162;
};
template<>
struct Qk_vec_<__nv_bfloat16, 64> {
    using Type = __nv_bfloat162;
};
template<>
struct Qk_vec_<__nv_bfloat16, 128> {
    using Type = bf16_4_t;
};
template<>
struct Qk_vec_<__nv_bfloat16, 256> {
    using Type = bf16_8_t;
};
#endif  // ENABLE_BF16
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int THREADS_PER_KEY>
struct K_vec_ {};

template<>
struct K_vec_<float, 4> {
    using Type = float;
};
template<>
struct K_vec_<float, 2> {
    using Type = float2;
};
template<>
struct K_vec_<float, 1> {
    using Type = float4;
};
template<>
struct K_vec_<uint16_t, 4> {
    using Type = uint32_t;
};
template<>
struct K_vec_<uint16_t, 2> {
    using Type = uint2;
};
template<>
struct K_vec_<uint16_t, 1> {
    using Type = uint4;
};
#ifdef ENABLE_BF16
template<>
struct K_vec_<__nv_bfloat16, 4> {
    using Type = __nv_bfloat162;
};
template<>
struct K_vec_<__nv_bfloat16, 2> {
    using Type = bf16_4_t;
};
template<>
struct K_vec_<__nv_bfloat16, 1> {
    using Type = bf16_8_t;
};
#endif  // ENABLE_BF16
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int V_VEC_SIZE>
struct V_vec_ {};

template<>
struct V_vec_<float, 1> {
    using Type = float;
};
template<>
struct V_vec_<float, 2> {
    using Type = float2;
};
template<>
struct V_vec_<float, 4> {
    using Type = float4;
};
template<>
struct V_vec_<uint16_t, 2> {
    using Type = uint32_t;
};
template<>
struct V_vec_<uint16_t, 4> {
    using Type = uint2;
};
template<>
struct V_vec_<uint16_t, 8> {
    using Type = uint4;
};
#ifdef ENABLE_BF16
template<>
struct V_vec_<__nv_bfloat16, 2> {
    using Type = __nv_bfloat162;
};
template<>
struct V_vec_<__nv_bfloat16, 4> {
    using Type = bf16_4_t;
};
template<>
struct V_vec_<__nv_bfloat16, 8> {
    using Type = bf16_8_t;
};
#endif  // ENABLE_BF16
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
template<typename T>
struct Qk_vec_acum_fp32_ {};

template<>
struct Qk_vec_acum_fp32_<float> {
    using Type = float;
};
template<>
struct Qk_vec_acum_fp32_<float2> {
    using Type = float2;
};
template<>
struct Qk_vec_acum_fp32_<float4> {
    using Type = float4;
};
// template<> struct Qk_vec_acum_fp32_<uint16_t> { using Type = float;        };
template<>
struct Qk_vec_acum_fp32_<uint32_t> {
    using Type = float2;
};
template<>
struct Qk_vec_acum_fp32_<uint2> {
    using Type = Float4_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct K_vec_acum_fp32_ {};

template<>
struct K_vec_acum_fp32_<float> {
    using Type = float;
};
template<>
struct K_vec_acum_fp32_<float2> {
    using Type = float2;
};
template<>
struct K_vec_acum_fp32_<float4> {
    using Type = float4;
};
template<>
struct K_vec_acum_fp32_<uint32_t> {
    using Type = float2;
};
template<>
struct K_vec_acum_fp32_<uint2> {
    using Type = Float4_;
};
template<>
struct K_vec_acum_fp32_<uint4> {
    using Type = Float8_;
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
template<typename T>
struct V_vec_acum_fp32_ {};

template<>
struct V_vec_acum_fp32_<float> {
    using Type = float;
};
template<>
struct V_vec_acum_fp32_<float2> {
    using Type = float2;
};
template<>
struct V_vec_acum_fp32_<float4> {
    using Type = float4;
};
template<>
struct V_vec_acum_fp32_<uint32_t> {
    using Type = float2;
};
template<>
struct V_vec_acum_fp32_<uint2> {
    using Type = Float4_;
};
template<>
struct V_vec_acum_fp32_<uint4> {
    using Type = Float8_;
};
#ifdef ENABLE_BF16
template<>
struct V_vec_acum_fp32_<__nv_bfloat162> {
    using Type = float2;
};
template<>
struct V_vec_acum_fp32_<bf16_4_t> {
    using Type = Float4_;
};
template<>
struct V_vec_acum_fp32_<bf16_8_t> {
    using Type = Float8_;
};
#endif  // ENABLE_BF16
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS_PER_KEY, typename K_vec, int N>
inline __device__ float qk_dot_(const K_vec (&q)[N], const K_vec (&k)[N])
{
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
    using K_vec_acum = typename K_vec_acum_fp32_<K_vec>::Type;
#else
    using K_vec_acum = K_vec;
#endif
    // Compute the parallel products for Q*K^T (treat vector lanes separately).
    K_vec_acum qk_vec = mul<K_vec_acum, K_vec, K_vec>(q[0], k[0]);
#pragma unroll
    for (int ii = 1; ii < N; ++ii) {
        qk_vec = fma(q[ii], k[ii], qk_vec);
    }

    // Finalize the reduction across lanes.
    float qk = sum(qk_vec);
#pragma unroll
    for (int mask = THREADS_PER_KEY / 2; mask >= 1; mask /= 2) {
        qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
    }
    return qk;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int THREADS_PER_KEY>
struct Qk_dot {
    template<typename K_vec, int N>
    static inline __device__ float dot(const K_vec (&q)[N], const K_vec (&k)[N])
    {
        return qk_dot_<THREADS_PER_KEY>(q, k);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 hmma_fp32(const uint2& a, uint32_t b)
{
    float4 c;
    float zero = 0.f;
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \n"
                 "    {%0, %1, %2, %3}, \n"
                 "    {%4, %5}, \n"
                 "    {%6}, \n"
                 "    {%7, %7, %7, %7}; \n"

                 : "=f"(c.x), "=f"(c.y), "=f"(c.z), "=f"(c.w)
                 : "r"(a.x) "r"(a.y), "r"(b), "f"(zero));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int N>
inline __device__ float qk_hmma_dot_(const uint32_t (&q)[N], const uint32_t (&k)[N])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
    using K_vec_acum = typename K_vec_acum_fp32_<uint32_t>::Type;
#else
    using K_vec_acum = uint32_t;
#endif
    K_vec_acum qk_vec = mul<K_vec_acum, uint32_t, uint32_t>(q[0], k[0]);
#pragma unroll
    for (int ii = 1; ii < N; ++ii) {
        qk_vec = fma(q[ii], k[ii], qk_vec);
    }
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
    uint32_t qk_vec_ = float2_to_half2(qk_vec);
    return hmma_fp32(make_uint2(qk_vec_, 0u), 0x3c003c00u).x;
#else
    return hmma_fp32(make_uint2(qk_vec, 0u), 0x3c003c00u).x;
#endif
#else
    return 0.f;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Qk_dot<uint16_t, 4> {
    template<int N>
    static inline __device__ float dot(const uint32_t (&q)[N], const uint32_t (&k)[N])
    {
#if __CUDA_ARCH__ >= 750 && defined(MMHA_USE_HMMA_FOR_REDUCTION)
        return qk_hmma_dot_(q, k);
#else
        return qk_dot_<4>(q, k);
#endif  // defined MMHA_USE_HMMA_FOR_REDUCTION
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int WARPS_PER_BLOCK, int WARP_SIZE = 32>
inline __device__ float block_sum(float* red_smem, float sum)
{

    // Decompose the thread index into warp / lane.
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

// Compute the sum per warp.
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
    }

    // Warp leaders store the data to shared memory.
    if (lane == 0) {
        red_smem[warp] = sum;
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // The warps compute the final sums.
    if (lane < WARPS_PER_BLOCK) {
        sum = red_smem[lane];
    }

// Parallel reduction inside the warp.
#pragma unroll
    for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
        sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
    }

    // Broadcast to other threads.
    return __shfl_sync(uint32_t(-1), sum, 0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(float& dst, float src)
{
    dst = src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint16_t& dst, float src)
{
    dst = float_to_half(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint32_t& dst, float2 src)
{
    dst = float2_to_half2(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef ENABLE_BF16
inline __device__ void convert_from_float(__nv_bfloat16& dst, float src)
{
    dst = __float2bfloat16(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(__nv_bfloat162& dst, float2 src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    dst = __float22bfloat162_rn(src);
#else
    dst = __floats2bfloat162_rn(src.x, src.y);
#endif
}
#endif  // ENABLE_BF16
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint2& dst, Float4_ src)
{
    dst.x = float2_to_half2(src.x);
    dst.y = float2_to_half2(src.y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint4& dst, Float8_ src)
{
    dst.x = float2_to_half2(src.x);
    dst.y = float2_to_half2(src.y);
    dst.z = float2_to_half2(src.z);
    dst.w = float2_to_half2(src.w);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16
inline __device__ void convert_from_float(bf16_4_t& dst, Float4_ src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    dst.x = __float22bfloat162_rn(src.x);
    dst.y = __float22bfloat162_rn(src.y);
#else
    dst.x = __floats2bfloat162_rn(src.x.x, src.x.y);
    dst.y = __floats2bfloat162_rn(src.y.x, src.y.y);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(bf16_8_t& dst, Float8_ src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    dst.x = __float22bfloat162_rn(src.x);
    dst.y = __float22bfloat162_rn(src.y);
    dst.z = __float22bfloat162_rn(src.z);
    dst.w = __float22bfloat162_rn(src.w);
#else
    dst.x = __floats2bfloat162_rn(src.x.x, src.x.y);
    dst.y = __floats2bfloat162_rn(src.y.x, src.y.y);
    dst.z = __floats2bfloat162_rn(src.z.x, src.z.y);
    dst.w = __floats2bfloat162_rn(src.w.x, src.w.y);
#endif
}
#endif  // ENABLE_BF16

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(float2& dst, float2 src)
{
    dst = src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(float4& dst, float4 src)
{
    dst = src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float convert_to_float(float4 u)
{
    return u.x;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float convert_to_float(uint4 u)
{
    float2 tmp = half2_to_float2(u.x);
    return tmp.x;
}

#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float cast_to_float(float u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 cast_to_float(float2 u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 cast_to_float(float4 u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ cast_to_float(Float4_ u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ cast_to_float(Float8_ u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 cast_to_float(uint32_t u)
{
    return half2_to_float2(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ cast_to_float(uint2 u)
{
    Float4_ tmp;
    tmp.x = half2_to_float2(u.x);
    tmp.y = half2_to_float2(u.y);
    return tmp;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ cast_to_float(uint4 u)
{
    Float8_ tmp;
    tmp.x = half2_to_float2(u.x);
    tmp.y = half2_to_float2(u.y);
    tmp.z = half2_to_float2(u.z);
    tmp.w = half2_to_float2(u.w);
    return tmp;
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline __device__ __host__ T div_up(T m, T n)
{
    return (m + n - 1) / n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, bool DO_CROSS_ATTENTION>
inline size_t smem_size_in_bytes(const Multihead_attention_params<T, DO_CROSS_ATTENTION>& params,
                                 int threads_per_value,
                                 int threads_per_block)
{
    // The amount of shared memory needed to store the Q*K^T values in float.
    // TODO
    size_t qk_sz = (DO_CROSS_ATTENTION) ? div_up(params.seq_length + 1, 4) * 16 : div_up(params.timestep + 1, 4) * 16;

    // The extra memory needed if we are not using floats for the final logits.
    size_t logits_sz = 0;
#ifndef MMHA_USE_FP32_ACUM_FOR_LOGITS
    if (sizeof(T) != 4) {
        // TDOD
        logits_sz = div_up(params.seq_length, 4) * 4 * sizeof(T);
    }
#endif

    // The total size needed during softmax.
    size_t softmax_sz = qk_sz + logits_sz;

    // The number of partial rows to reduce in the final reduction.
    int rows_per_red = threads_per_block / threads_per_value;
    // The amount of storage needed to finalize the outputs.
    size_t red_sz = rows_per_red * params.hidden_size_per_head * sizeof(T) / 2;

    // The max.
    return max(softmax_sz, red_sz);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ constexpr uint32_t shfl_mask(int threads)
{
    return threads == 32 ? uint32_t(-1) : (1u << threads) - 1u;
}


template<
    // The type of the inputs. Supported types: float and half.
    typename T,
    // The hidden dimension per head.
    int Dh,
    int Dh_MAX,
    // The number of threads per key.
    int THREADS_PER_KEY,
    // The number of threads per value.
    int THREADS_PER_VALUE,
    // The number of threads in a threadblock.
    int THREADS_PER_BLOCK,
    bool DO_CROSS_ATTENTION,
    bool skip_mem, bool skip_loop>
__global__ void masked_multihead_attention_kernel_optimized(Multihead_attention_params<T, DO_CROSS_ATTENTION> params)
{
    // Make sure the hidden dimension per head is a multiple of the number of threads per key.
    static_assert(Dh_MAX % THREADS_PER_KEY == 0, "");
    // Make sure the hidden dimension per head is a multiple of the number of threads per value.
    static_assert(Dh_MAX % THREADS_PER_VALUE == 0, "");

    // The size of a warp.
    constexpr int WARP_SIZE = 32;
    // The number of warps in a threadblock.
    constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

    // Use smem_size_in_bytes (above) to determine the amount of shared memory.
    extern __shared__ char smem_[];

    // The shared memory for the Q*K^T values and partial logits in softmax.
    float* qk_smem = reinterpret_cast<float*>(smem_);

    // The shared memory for the logits. For FP32, that's the same buffer as qk_smem.
    char* logits_smem_ = smem_;
#ifndef MMHA_USE_FP32_ACUM_FOR_LOGITS
    if (sizeof(T) != 4) {
        // TODO - cahnge to tlength
        logits_smem_ +=
            (DO_CROSS_ATTENTION) ? div_up(params.seq_length + 1, 4) * 16 : div_up(params.timestep + 1, 4) * 16;
    }
    T* logits_smem = reinterpret_cast<T*>(logits_smem_);
#else
    float* logits_smem = reinterpret_cast<float*>(logits_smem_);
#endif

    // The shared memory to do the final reduction for the output values. Reuse qk_smem.
    T* out_smem = reinterpret_cast<T*>(smem_);

    // The shared memory buffers for the block-wide reductions. One for max, one for sum.
    __shared__ float red_smem[WARPS_PER_BLOCK * 2];

    // A vector of Q or K elements for the current timestep.
    using Qk_vec = typename Qk_vec_<T, Dh_MAX>::Type;

    // Use alignment for safely casting the shared buffers as Qk_vec.
    // Shared memory to store Q inputs.
    __shared__ __align__(sizeof(Qk_vec)) T q_smem[Dh_MAX];

    // This is one of the reasons we should have a separate kernel for cross attention
    __shared__ __align__(sizeof(Qk_vec)) T bias_smem[DO_CROSS_ATTENTION ? Dh_MAX : 1];

    // A vector of Q or K elements for the current timestep.
    using Qk_vec = typename Qk_vec_<T, Dh_MAX>::Type;
    // The number of elements per vector.
    constexpr int QK_VEC_SIZE = sizeof(Qk_vec) / sizeof(T);
    // Make sure the hidden size per head is a multiple of the vector size.
    static_assert(Dh_MAX % QK_VEC_SIZE == 0, "");
    // We will use block wide reduction if needed
    // static_assert(Dh_MAX / QK_VEC_SIZE <= WARP_SIZE, "");
    // The number of vectors per warp.
    constexpr int QK_VECS_PER_WARP = Dh_MAX / QK_VEC_SIZE;

    // The layout of the cache is [B, H, Dh/x, L, x] with x == 4/8 for FP32/FP16. Since each thread
    // owns x elements, we have to decompose the linear index into chunks of x values and the posi-
    // tion of the thread in that chunk.

    // The number of elements in a chunk of 16B (that's the x in the above formula).
    constexpr int QK_ELTS_IN_16B = 16 / sizeof(T);
    // The number of K vectors in 16B.
    constexpr int QK_VECS_IN_16B = 16 / sizeof(Qk_vec);

    // The batch/beam idx
    const int bi = blockIdx.y;
    if (params.finished != nullptr && params.finished[bi] == true) {
        return;
    }
    // The beam idx
    const int beami = bi % params.beam_width;
    // The "beam-aware" batch idx
    const int bbi = bi / params.beam_width;
    // The head.
    const int hi = blockIdx.x;
    // Combine the batch and the head indices.
    const int bhi = bi * params.num_heads + hi;
    // Combine the "beam-aware" batch idx and the head indices.
    const int bbhi = bbi * params.beam_width * params.num_heads + hi;
    // The thread in the block.
    const int tidx = threadIdx.x;

    // While doing the product Q*K^T for the different keys we track the max.
    float qk_max = -FLT_MAX;

    float qk = 0.0F;

    int qkv_base_offset = (params.stride == 0) ? bhi * Dh : bi * params.stride + hi * Dh;

    // int tlength = (DO_CROSS_ATTENTION)? params.memory_length_per_sample[bi] - 1 : params.timestep;
    int tlength = (DO_CROSS_ATTENTION)                  ? params.memory_length_per_sample[bi] - 1 :
                  (params.length_per_sample == nullptr) ? params.timestep :
                                                          params.length_per_sample[bi];

    T* dummy_qkvec[QK_VEC_SIZE];
    // First QK_VECS_PER_WARP load Q and K + the bias values for the current timestep.
    if (tidx < QK_VECS_PER_WARP) {

        // The offset in the Q and K buffer also accounts for the batch.
        int qk_offset = qkv_base_offset + tidx * QK_VEC_SIZE;
        // The offset in the bias buffer.
        int qk_bias_offset = hi * Dh + tidx * QK_VEC_SIZE;

        // Trigger the loads from the Q and K buffers.
        Qk_vec q;

        zero(q);
        if (!skip_mem)
            q = (Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh) ? *reinterpret_cast<const Qk_vec*>(&params.q[qk_offset]) : q;
        else
            q = (Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh) ? *reinterpret_cast<const Qk_vec*>(dummy_qkvec) : q;

        Qk_vec k;
        zero(k);
        if (DO_CROSS_ATTENTION) {
            // The 16B chunk written by the thread.
            int co = tidx / QK_VECS_IN_16B;
            // The position of the thread in that 16B chunk.
            int ci = tidx % QK_VECS_IN_16B * QK_VEC_SIZE;

            // Two chunks are separated by L * x elements. A thread write QK_VEC_SIZE elements.
            int offset = bhi * params.seq_length * Dh + co * params.seq_length * QK_ELTS_IN_16B +
                         // params.timestep*QK_ELTS_IN_16B +
                         tlength * QK_ELTS_IN_16B + ci;
            k = (Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh) ? *reinterpret_cast<const Qk_vec*>(&params.k_cache[offset]) :
                                                            k;
        }
        else {
            if (!skip_mem)
                k = (Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh) ? *reinterpret_cast<const Qk_vec*>(&params.k[qk_offset]) : k;
            else
                k = (Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh) ? *reinterpret_cast<const Qk_vec*>(dummy_qkvec) : k;
        }

        // Trigger the loads from the Q and K bias buffers.
        Qk_vec q_bias;
        zero(q_bias);
        q_bias = (Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh) && params.q_bias != nullptr ?
                     *reinterpret_cast<const Qk_vec*>(&params.q_bias[qk_bias_offset]) :
                     q_bias;
        Qk_vec k_bias;
        zero(k_bias);

        if (!DO_CROSS_ATTENTION || (DO_CROSS_ATTENTION && params.timestep == 0)) {
            k_bias = (Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh) && params.k_bias != nullptr ?
                         *reinterpret_cast<const Qk_vec*>(&params.k_bias[qk_bias_offset]) :
                         k_bias;
        }

        // Computes the Q/K values with bias.
        q = add(q, q_bias);
        if (!DO_CROSS_ATTENTION || (DO_CROSS_ATTENTION && params.timestep == 0)) {
            k = add(k, k_bias);
            if (params.rotary_embedding_dim > 0) {
                apply_rotary_embedding(q, k, tidx, params.rotary_embedding_dim, params.timestep);
            }
        }
        else {
            if (params.rotary_embedding_dim > 0) {
                apply_rotary_embedding(q, tidx, params.rotary_embedding_dim, params.timestep);
            }
        }

        // Store the Q values to shared memory.
        *reinterpret_cast<Qk_vec*>(&q_smem[tidx * QK_VEC_SIZE]) = q;

        // Store Dh values of k_bias into smem, since will need to add later
        // if params.timestep == 0
        if (DO_CROSS_ATTENTION && params.timestep == 0) {
            *reinterpret_cast<Qk_vec*>(&bias_smem[tidx * QK_VEC_SIZE]) = k_bias;
        }

        // Write the K values to the global memory cache.
        //
        // NOTE: The stores are uncoalesced as we have multiple chunks of 16B spread across the memory
        // system. We designed it this way as it allows much better memory loads (and there are many
        // more loads) + the stores are really "write and forget" since we won't need the ack before
        // the end of the kernel. There's plenty of time for the transactions to complete.

        // The 16B chunk written by the thread.
        int co = tidx / QK_VECS_IN_16B;
        // The position of the thread in that 16B chunk.
        int ci = tidx % QK_VECS_IN_16B * QK_VEC_SIZE;

        // Two chunks are separated by L * x elements. A thread write QK_VEC_SIZE elements.
        int offset = bhi * params.seq_length * Dh + co * params.seq_length * QK_ELTS_IN_16B +
                     // params.timestep*QK_ELTS_IN_16B +
                     tlength * QK_ELTS_IN_16B + ci;

        if (!DO_CROSS_ATTENTION || (DO_CROSS_ATTENTION && params.timestep == 0)) {
            // Trigger the stores to global memory.
            if (Dh == Dh_MAX || co < Dh / QK_ELTS_IN_16B) {
                if (!skip_mem)
                    *reinterpret_cast<Qk_vec*>(&params.k_cache[offset]) = k;
            }
        }

        // Compute \sum_i Q[i] * K^T[i] for the current timestep.
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
        using Qk_vec_acum = typename Qk_vec_acum_fp32_<Qk_vec>::Type;
#else
        using Qk_vec_acum = Qk_vec;
#endif
        qk = dot<Qk_vec_acum, Qk_vec>(q, k);
        if (QK_VECS_PER_WARP <= WARP_SIZE) {
#pragma unroll
            for (int mask = QK_VECS_PER_WARP / 2; mask >= 1; mask /= 2) {
                qk += __shfl_xor_sync(shfl_mask(QK_VECS_PER_WARP), qk, mask);
            }
        }
    }

    if (QK_VECS_PER_WARP > WARP_SIZE) {
        constexpr int WARPS_PER_RED = (QK_VECS_PER_WARP + WARP_SIZE - 1) / WARP_SIZE;
        qk = block_sum<WARPS_PER_RED>(&red_smem[WARPS_PER_RED], qk);
    }

    // Store that value in shared memory. Keep the Q*K^T value in register for softmax.
    if (tidx == 0) {
        // Normalize qk.
        qk *= params.inv_sqrt_dh;

        if (params.relative_attention_bias_float != nullptr) {
            qk = qk
                 + params.relative_attention_bias_float[hi * params.relative_attention_bias_stride
                                                            * params.relative_attention_bias_stride
                                                        + tlength * params.relative_attention_bias_stride + tlength];
        }
        else if (params.relative_attention_bias_half != nullptr) {
            qk = qk
                 + (float)
                       params.relative_attention_bias_half[hi * params.relative_attention_bias_stride
                                                               * params.relative_attention_bias_stride
                                                           + tlength * params.relative_attention_bias_stride + tlength];
        }
        qk_max = qk;
        qk_smem[tlength] = qk;
        // qk_smem[params.timestep] = qk;
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // The type of queries and keys for the math in the Q*K^T product.
    using K_vec = typename K_vec_<T, THREADS_PER_KEY>::Type;
    // The number of elements per vector.
    constexpr int K_VEC_SIZE = sizeof(K_vec) / sizeof(T);
    // Make sure the hidden size per head is a multiple of the vector size.
    static_assert(Dh_MAX % K_VEC_SIZE == 0, "");
    // The number of elements per thread.
    constexpr int K_ELTS_PER_THREAD = Dh_MAX / THREADS_PER_KEY;
    // The number of vectors per thread.
    constexpr int K_VECS_PER_THREAD = K_ELTS_PER_THREAD / K_VEC_SIZE;

    // The position the first key loaded by each thread from the cache buffer (for this B * H).
    int ko = tidx / THREADS_PER_KEY;
    // The position of the thread in the chunk of keys.
    int ki = tidx % THREADS_PER_KEY * K_VEC_SIZE;

    static_assert(Dh_MAX == THREADS_PER_KEY * K_VEC_SIZE * K_VECS_PER_THREAD);

    // Load the Q values from shared memory. The values are reused during the loop on K.
    K_vec q[K_VECS_PER_THREAD];
#pragma unroll
    for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
        q[ii] = *reinterpret_cast<const K_vec*>(&q_smem[ki + ii * THREADS_PER_KEY * K_VEC_SIZE]);
    }

    K_vec k_bias[DO_CROSS_ATTENTION ? K_VECS_PER_THREAD : 1];
    if (DO_CROSS_ATTENTION && params.timestep == 0) {
#pragma unroll
        for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
            k_bias[ii] = *reinterpret_cast<const K_vec*>(&bias_smem[ki + ii * THREADS_PER_KEY * K_VEC_SIZE]);
        }
    }

    // The number of timesteps loaded per iteration.
    constexpr int K_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_KEY;
    // The number of keys per warp.
    constexpr int K_PER_WARP = WARP_SIZE / THREADS_PER_KEY;

    // The base pointer for the key in the cache buffer.
    T* k_cache = &params.k_cache[bhi * params.seq_length * Dh + ki];
    // Base pointer for the beam's batch, before offsetting with indirection buffer
    T* k_cache_batch = &params.k_cache[bbhi * params.seq_length * Dh + ki];

    // Pick a number of keys to make sure all the threads of a warp enter (due to shfl_sync).
    // int ti_end = div_up(params.timestep, K_PER_WARP) * K_PER_WARP;
    int ti_end = div_up(tlength, K_PER_WARP) * K_PER_WARP;
    

    T* dummy_kvec[K_VEC_SIZE];

    // Iterate over the keys/timesteps to compute the various (Q*K^T)_{ti} values.
    if (!skip_loop){
        for (int ti = ko; ti < ti_end; ti += K_PER_ITER) {

            // The keys loaded from the key cache.
            K_vec k[K_VECS_PER_THREAD];
            K_vec k_vec_zero;
            zero(k_vec_zero);
#pragma unroll
            for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
                int jj = ii * params.seq_length + ti;
                // if( ti < params.timestep ) {
                if (ti < tlength) {
                    if (!skip_mem){
                        if (params.cache_indir != nullptr){
                            const int beam_src =
                                (params.cache_indir != nullptr) ?
                                    params.cache_indir[(bbi * params.beam_width + beami) * params.seq_length + ti] :
                                    0;
                            const int beam_offset = beam_src * params.num_heads * params.seq_length * Dh;
                            k[ii] = (Dh == Dh_MAX || jj * QK_ELTS_IN_16B < Dh * params.seq_length) ?
                                        *reinterpret_cast<const K_vec*>(&k_cache_batch[beam_offset + jj * QK_ELTS_IN_16B]) :
                                        k_vec_zero;
                        }
                        else{
                            k[ii] = (Dh == Dh_MAX || jj * QK_ELTS_IN_16B < Dh * params.seq_length) ?
                                        *reinterpret_cast<const K_vec*>(&k_cache_batch[0 + jj * QK_ELTS_IN_16B]) :
                                        k_vec_zero;
                        }
                        // add bias and update k_cache
                    }
                    else{
                        k[ii] = (Dh == Dh_MAX || jj * QK_ELTS_IN_16B < Dh * params.seq_length) ?
                                    *reinterpret_cast<const K_vec*>(dummy_kvec) :
                                    k_vec_zero;
                    }

                    if (DO_CROSS_ATTENTION && params.timestep == 0) {
                        k[ii] = add(k[ii], k_bias[ii]);
                        if (Dh == Dh_MAX || jj * QK_ELTS_IN_16B < Dh * params.seq_length) {
                            *reinterpret_cast<K_vec*>(&k_cache[jj * QK_ELTS_IN_16B]) = k[ii];
                        }
                    }
                }
            }

            // Perform the dot product and normalize qk.
            //
            // WARNING: ALL THE THREADS OF A WARP MUST ENTER!!!
            float qk = Qk_dot<T, THREADS_PER_KEY>::dot(q, k) * params.inv_sqrt_dh;
            bool is_mask = (params.input_lengths != nullptr && ti >= params.input_lengths[bi] && ti < params.max_input_len);

            // Store the product to shared memory. There's one qk value per timestep. Update the max.
            // if( ti < params.timestep && tidx % THREADS_PER_KEY == 0 ) {
            if (ti < tlength && tidx % THREADS_PER_KEY == 0) {
                if (params.relative_attention_bias_float != nullptr) {
                    qk = qk
                         + params.relative_attention_bias_float[hi * params.relative_attention_bias_stride
                                                                    * params.relative_attention_bias_stride
                                                                + tlength * params.relative_attention_bias_stride + ti];
                }
                else if (params.relative_attention_bias_half != nullptr) {
                    qk = qk
                         + (float)
                               params.relative_attention_bias_half[hi * params.relative_attention_bias_stride
                                                                       * params.relative_attention_bias_stride
                                                                   + tlength * params.relative_attention_bias_stride + ti];
                }
                qk_max = is_mask ? qk_max : fmaxf(qk_max, qk);
                qk_smem[ti] = qk;
            }
        }
    }

// Perform the final reduction to compute the max inside each warp.
//
// NOTE: In a group of THREADS_PER_KEY threads, the leader already has the max value for the
// group so it's not needed to run the reduction inside the group (again).
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= THREADS_PER_KEY; mask /= 2) {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
    }

    // Decompose the thread index into warp and lane.
    const int warp = tidx / WARP_SIZE;
    const int lane = tidx % WARP_SIZE;

    // The warp leader writes the max to shared memory.
    if (lane == 0) {
        red_smem[warp] = qk_max;
    }

    // Make sure the products are in shared memory.
    __syncthreads();

    // The warps finalize the reduction.
    qk_max = lane < WARPS_PER_BLOCK ? red_smem[lane] : -FLT_MAX;
#pragma unroll
    for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
    }

    // Broadcast to all the threads in the warp.
    qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

    // Compute the logits and start the sum.
    float sum = 0.f;
    // for( int ti = tidx; ti <= params.timestep; ti += THREADS_PER_BLOCK ) {
    if (!skip_loop){
        for (int ti = tidx; ti <= tlength; ti += THREADS_PER_BLOCK) {
            bool is_mask = (params.input_lengths != nullptr && ti >= params.input_lengths[bi] && ti < params.max_input_len);
            float logit = is_mask ? 0.f : __expf(qk_smem[ti] - qk_max);
            sum += logit;
            qk_smem[ti] = logit;
        }
    }

    // Compute the sum.
    sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], sum);

    // Normalize the logits.
    float inv_sum = __fdividef(1.f, sum + 1.e-6f);
    // for( int ti = tidx; ti <= params.timestep; ti += THREADS_PER_BLOCK ) {
    if (!skip_loop){
        for (int ti = tidx; ti <= tlength; ti += THREADS_PER_BLOCK) {
            convert_from_float(logits_smem[ti], qk_smem[ti] * inv_sum);
        }
    }

    // Put Values part below so we leverage __syncthreads
    // from the previous step

    // The number of elements per vector.
    constexpr int V_VEC_SIZE = Dh_MAX / THREADS_PER_VALUE;
    // A vector of V elements for the current timestep.
    using V_vec = typename V_vec_<T, V_VEC_SIZE>::Type;

    // The value computed by this thread.
    int vo = tidx / THREADS_PER_VALUE;
    // The hidden dimensions computed by this particular thread.
    int vi = tidx % THREADS_PER_VALUE * V_VEC_SIZE;

    // The base pointer for the value in the cache buffer.
    T* v_cache = &params.v_cache[bhi * params.seq_length * Dh + vi];
    // Base pointer for the beam's batch, before offsetting with indirection buffer
    T* v_cache_batch = &params.v_cache[bbhi * params.seq_length * Dh + vi];

    // The number of values processed per iteration of the loop.
    constexpr int V_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;

    // One group of threads computes the product(s) for the current timestep.
    V_vec v_bias;
    zero(v_bias);
    // if( vo == params.timestep % V_PER_ITER ) {
    if (Dh == Dh_MAX || vi < Dh) {
        if (!DO_CROSS_ATTENTION || (DO_CROSS_ATTENTION && params.timestep == 0)) {
            if (vo == tlength % V_PER_ITER) {
                // Trigger the loads from the V bias buffer.
                if (params.v_bias != nullptr) {
                    v_bias = *reinterpret_cast<const V_vec*>(&params.v_bias[hi * Dh + vi]);
                }
                if (DO_CROSS_ATTENTION) {
                    *reinterpret_cast<V_vec*>(&bias_smem[vi]) = v_bias;
                }
            }
        }
    }

    // From previous, before values, step
    // Also make sure the logits are in shared memory.
    __syncthreads();

    // Values continued
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
    using V_vec_acum = typename V_vec_acum_fp32_<V_vec>::Type;
#else
    using V_vec_acum = V_vec;
#endif
    // The partial outputs computed by each thread.
    V_vec_acum out;
    zero(out);

    T* dummy_vvec[V_VEC_SIZE];

    // Loop over the timesteps to compute the partial outputs.
    // for( int ti = vo; ti < params.timestep; ti += V_PER_ITER ) {
    if (!skip_loop){
        if (Dh == Dh_MAX || vi < Dh) {
            for (int ti = vo; ti < tlength; ti += V_PER_ITER) {

                // Fetch offset based on cache_indir when beam sampling
                V_vec v;
                if (!skip_mem){
                    if (params.cache_indir != nullptr){
                        const int beam_src = (params.cache_indir != nullptr) ?
                                                 params.cache_indir[(bbi * params.beam_width + beami) * params.seq_length + ti] :
                                                 0;
                        const int beam_offset = beam_src * params.num_heads * params.seq_length * Dh;
                        // Load the values from the cache.
                        v = *reinterpret_cast<const V_vec*>(&v_cache_batch[beam_offset + ti * Dh]);
                    }
                    else{
                        v = *reinterpret_cast<const V_vec*>(&v_cache_batch[0 + ti * Dh]);
                    }
                }
                else{
                    v = *reinterpret_cast<const V_vec*>(dummy_vvec);
                }


                if (DO_CROSS_ATTENTION && params.timestep == 0) {
                    v = add(v, *reinterpret_cast<V_vec*>(&bias_smem[vi]));
                    *reinterpret_cast<V_vec*>(&v_cache[ti * Dh]) = v;
                }
                // Load the logits from shared memory.
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
                float logit = logits_smem[ti];
                out = fma(logit, cast_to_float(v), out);
#else
                T logit = logits_smem[ti];

                // Update the partial sums.
                out = fma(logit, v, out);
#endif
            }
        }

        // One group of threads computes the product(s) for the current timestep.
        // if( vo == params.timestep % V_PER_ITER ) {
        if (vo == tlength % V_PER_ITER && (Dh == Dh_MAX || vi < Dh)) {

            V_vec v;
            if (DO_CROSS_ATTENTION) {
                v = *reinterpret_cast<const V_vec*>(&v_cache[tlength * Dh]);
            }
            else {
                // Trigger the loads from the V buffer.
                if (!skip_mem)
                    v = *reinterpret_cast<const V_vec*>(&params.v[qkv_base_offset + vi]);
                else
                    v = *reinterpret_cast<const V_vec*>(dummy_vvec);
                // Trigger the loads from the V bias buffer.
                // V_vec v_bias = *reinterpret_cast<const V_vec*>(&params.v_bias[hi*Dh + vi]);
            }

            // Compute the V values with bias.
            if (!DO_CROSS_ATTENTION || (DO_CROSS_ATTENTION && params.timestep == 0)) {
                v = add(v, v_bias);

                // Store the values with bias back to global memory in the cache for V.
                //*reinterpret_cast<V_vec*>(&v_cache[params.timestep*Dh]) = v;
                *reinterpret_cast<V_vec*>(&v_cache[tlength * Dh]) = v;
            }

            // Initialize the output value with the current timestep.
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
            // out = fma(logits_smem[params.timestep], cast_to_float(v), out);
            out = fma(logits_smem[tlength], cast_to_float(v), out);
#else
            // out = fma(logits_smem[params.timestep], v, out);
            out = fma(logits_smem[tlength], v, out);
#endif
        }

        // Make sure we can start writing to shared memory.
        __syncthreads();

        // Run the final reduction amongst the different groups computing different partial outputs.
        if (Dh == Dh_MAX || vi < Dh)
#pragma unroll
            for (int active_groups = V_PER_ITER; active_groups >= 2; active_groups /= 2) {

                // The midpoint in the number of active groups.
                int midpoint = active_groups / 2;

                // The upper part of active threads store to shared memory.
                if (vo >= midpoint && vo < active_groups && (Dh == Dh_MAX || vi < Dh)) {
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
                    convert_from_float(*reinterpret_cast<V_vec*>(&out_smem[(vo - midpoint) * Dh + vi]), out);
#else
                    *reinterpret_cast<V_vec*>(&out_smem[(vo - midpoint) * Dh + vi]) = out;
#endif
                }
                __syncthreads();

                // The bottom warps update their values.
                if (vo < midpoint && (Dh == Dh_MAX || vi < Dh)) {
                    out = add(*reinterpret_cast<const V_vec*>(&out_smem[vo * Dh + vi]), out);
                }
                __syncthreads();
            }
    }

    // Output the final values.
    if (vo == 0 && (Dh == Dh_MAX || vi < Dh)) {
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
        convert_from_float(*reinterpret_cast<V_vec*>(&params.out[bhi * Dh + vi]), out);
#else
        *reinterpret_cast<V_vec*>(&params.out[bhi * Dh + vi]) = out;
#endif
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace mmha

#define MMHA_LAUNCH_KERNEL_OPTIMIZED(T, Dh, Dh_MAX, THDS_PER_KEY, THDS_PER_VALUE, THDS_PER_BLOCK, DO_CROSS_ATTENTION, stream)    \
    size_t smem_sz = mmha::smem_size_in_bytes<T, DO_CROSS_ATTENTION>(params, THDS_PER_VALUE, THDS_PER_BLOCK);          \
    dim3 grid(params.num_heads, params.batch_size);                                                                    \
    mmha::masked_multihead_attention_kernel_optimized<T,                                                                         \
                                            Dh,                                                                        \
                                            Dh_MAX,                                                                    \
                                            THDS_PER_KEY,                                                              \
                                            THDS_PER_VALUE,                                                            \
                                            THDS_PER_BLOCK,                                                            \
                                            DO_CROSS_ATTENTION, true, true><<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(params)


template<typename T, int Dh, int Dh_MAX, int TPK, int TPB, typename KERNEL_PARAMS_TYPE, int T_size>
struct mmha_lahnch_TPV{
    static void call(const KERNEL_PARAMS_TYPE& params, const cudaStream_t& stream, int v_vec_size)
    {
        throw std::exception(); 
    }
};

template<typename T, int Dh, int Dh_MAX, int TPK, int TPB, typename KERNEL_PARAMS_TYPE>
struct mmha_lahnch_TPV<T, Dh, Dh_MAX, TPK, TPB, KERNEL_PARAMS_TYPE, 4>{
    static void call(const KERNEL_PARAMS_TYPE& params, const cudaStream_t& stream, int v_vec_size)
    {
        const bool DO_CROSS_ATTENTION = false;

        switch (v_vec_size){
            case 1:
                {
                    constexpr int THREADS_PER_VALUE = std::min(TPB, Dh_MAX);
#ifdef tpv1
                    MMHA_LAUNCH_KERNEL_OPTIMIZED(T, Dh, Dh_MAX, TPK, THREADS_PER_VALUE, TPB, DO_CROSS_ATTENTION, stream);
#endif
                    break;
                }
            case 2:
                {
                    constexpr int THREADS_PER_VALUE = Dh_MAX / 2;
#ifdef tpv2
                    MMHA_LAUNCH_KERNEL_OPTIMIZED(T, Dh, Dh_MAX, TPK, THREADS_PER_VALUE, TPB, DO_CROSS_ATTENTION, stream);
#endif
                    break;
                }
            case 4:
                {
                    constexpr int THREADS_PER_VALUE = Dh_MAX / 4;
#ifdef tpv4
                    MMHA_LAUNCH_KERNEL_OPTIMIZED(T, Dh, Dh_MAX, TPK, THREADS_PER_VALUE, TPB, DO_CROSS_ATTENTION, stream);
#endif
                    break;
                }
            default:
                assert(false);
        } 
    }
};

template<typename T, int Dh, int Dh_MAX, int TPK, int TPB, typename KERNEL_PARAMS_TYPE>
struct mmha_lahnch_TPV<T, Dh, Dh_MAX, TPK, TPB, KERNEL_PARAMS_TYPE, 2>{
    static void call(const KERNEL_PARAMS_TYPE& params, const cudaStream_t& stream, int v_vec_size) 
    {
        const bool DO_CROSS_ATTENTION = false;

        switch (v_vec_size){
            case 2:
                {
                    constexpr int THREADS_PER_VALUE = std::min(TPB, Dh_MAX / 2);
#ifdef tpv2
                    MMHA_LAUNCH_KERNEL_OPTIMIZED(T, Dh, Dh_MAX, TPK, THREADS_PER_VALUE, TPB, DO_CROSS_ATTENTION, stream);
#endif  
                    break;
                }
            case 4:
                {
                    constexpr int THREADS_PER_VALUE = Dh_MAX / 4;
#ifdef tpv4
                    MMHA_LAUNCH_KERNEL_OPTIMIZED(T, Dh, Dh_MAX, TPK, THREADS_PER_VALUE, TPB, DO_CROSS_ATTENTION, stream);
#endif  
                    break;
                }
            case 8:
                {
                    constexpr int THREADS_PER_VALUE = Dh_MAX / 8;
#ifdef tpv8
                    MMHA_LAUNCH_KERNEL_OPTIMIZED(T, Dh, Dh_MAX, TPK, THREADS_PER_VALUE, TPB, DO_CROSS_ATTENTION, stream);
#endif  
                    break;
                }
            default:
                assert(false);
        } 
    }
};


template<typename T, int Dh, int Dh_MAX, int TPB, typename KERNEL_PARAMS_TYPE>
void mmha_launch_TPK(const KERNEL_PARAMS_TYPE& params, const cudaStream_t& stream, int v_vec_size, int k_vec_size)
{
    switch (k_vec_size){
        case 1:
#ifdef tpk1
            mmha_lahnch_TPV<T, Dh, Dh_MAX, 1, TPB, KERNEL_PARAMS_TYPE, sizeof(T)>::call(params, stream, v_vec_size);
#endif  
            break;
        case 2:
#ifdef tpk2
            mmha_lahnch_TPV<T, Dh, Dh_MAX, 2, TPB, KERNEL_PARAMS_TYPE, sizeof(T)>::call(params, stream, v_vec_size);
#endif  
            break;
        case 4:
#ifdef tpk4
            mmha_lahnch_TPV<T, Dh, Dh_MAX, 4, TPB, KERNEL_PARAMS_TYPE, sizeof(T)>::call(params, stream, v_vec_size);
#endif  
            break;
        default:
            assert(false);
    }
}

template<typename T, int Dh, int Dh_MAX, typename KERNEL_PARAMS_TYPE>
void mmha_launch(const KERNEL_PARAMS_TYPE& params, const cudaStream_t& stream, int v_vec_size, int k_vec_size, int threads_per_block)
{
    switch (threads_per_block){
        case 64:
#ifdef tpb64
            mmha_launch_TPK<T, Dh, Dh_MAX, 64, KERNEL_PARAMS_TYPE>(params, stream, v_vec_size, k_vec_size);
#endif  
            break;
        case 128:
#ifdef tpb128
            mmha_launch_TPK<T, Dh, Dh_MAX, 128, KERNEL_PARAMS_TYPE>(params, stream, v_vec_size, k_vec_size);
#endif  
            break;
        case 256:
#ifdef tpb256
            mmha_launch_TPK<T, Dh, Dh_MAX, 256, KERNEL_PARAMS_TYPE>(params, stream, v_vec_size, k_vec_size);
#endif  
            break;
        default:
            assert(false);
    }
}

inline bool cuda_check_error(std::string message){
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA %s error: %s\n", message.c_str(), cudaGetErrorString(error));
        return false;
    }
    return true;
}

template<typename T>
void init_val(T* cu_array, int length)
{
    T* array =  (T*)malloc(sizeof(T)*length);
    for (int j = 0; j < length; j++){
        array[j] = (float)rand()/(float)(RAND_MAX);
    }
    cudaMemcpy(cu_array, array, sizeof(T) * length, cudaMemcpyHostToDevice);
    cuda_check_error("rand_init");
    free(array);
}

template<typename T, typename KERNEL_PARAMS_TYPE>
void reset_device_res(const KERNEL_PARAMS_TYPE& params, T** context_buf, int no_layers)
{
    for (int i = 0; i < no_layers; i++){
        init_val<T>(context_buf[i], params.batch_size * params.num_heads * params.hidden_size_per_head);
    }
}

template<typename T>
void free_mem(int no_layers, T** qkv_buf, T** key_cache, T** value_cache, T** context_buf, T** true_res_host, T** res_host)
{

    for (int i = 0; i < no_layers; i++){
        cudaFree((qkv_buf)[i]);
        cudaFree((context_buf)[i]);
        cudaFree((key_cache)[i]);
        cudaFree((value_cache)[i]);
    
        free(true_res_host[i]);
        free(res_host[i]);
    }

    delete qkv_buf;
    delete key_cache;
    delete value_cache;
    delete context_buf;
    delete true_res_host;
    delete res_host;
}


template<typename T, typename KERNEL_PARAMS_TYPE>
void allocate_mem(const KERNEL_PARAMS_TYPE& params, int no_layers, T*** qkv_buf, T*** key_cache, T*** value_cache, T*** context_buf, T*** true_res_host, T*** res_host)
{
    *qkv_buf = new T*[no_layers];
    *key_cache = new T*[no_layers];
    *value_cache = new T*[no_layers];
    *context_buf = new T*[no_layers];
    (*true_res_host) = new T*[no_layers];
    (*res_host) = new T*[no_layers];

    for (int i = 0; i < no_layers; i++){
        cudaMalloc(&(*qkv_buf)[i], sizeof(T) * params.batch_size * 3 * params.num_heads*params.hidden_size_per_head);
        cudaMalloc(&(*context_buf)[i], sizeof(T)* params.batch_size * params.num_heads * params.hidden_size_per_head);
        cudaMalloc(&(*key_cache)[i], sizeof(T)*params.batch_size*params.num_heads * params.hidden_size_per_head*params.seq_length);
        cudaMalloc(&(*value_cache)[i], sizeof(T)*params.batch_size*params.num_heads * params.hidden_size_per_head*params.seq_length);

        init_val<T>((*qkv_buf)[i], params.batch_size * 3 * params.num_heads*params.hidden_size_per_head);
        init_val<T>((*context_buf)[i], params.batch_size * params.num_heads * params.hidden_size_per_head);
        init_val<T>((*key_cache)[i], params.batch_size*params.num_heads * params.hidden_size_per_head*params.seq_length);
        init_val<T>((*value_cache)[i], params.batch_size*params.num_heads * params.hidden_size_per_head*params.seq_length);

        (*true_res_host)[i] = (T*)malloc(sizeof(T)* params.batch_size * params.num_heads * params.hidden_size_per_head);
        (*res_host)[i] = (T*)malloc(sizeof(T)* params.batch_size * params.num_heads * params.hidden_size_per_head);
        for (int j = 0; j < params.batch_size * params.num_heads*params.hidden_size_per_head; j++){
            (*true_res_host)[i][j] = 5.0;
            (*res_host)[i][j] = 9.0;
        }
    }
}



template<typename T, int Dh, int Dh_MAX, typename KERNEL_PARAMS_TYPE>
bool forward(KERNEL_PARAMS_TYPE params, const cudaStream_t& stream, int v_vec_size, int k_vec_size, int threads_per_block, int no_layers, T** qkv_buf, T** key_cache, T** value_cache, T** context_buf, bool default_launch, T** res_host=nullptr){
    for (int i = params.max_input_len; i < params.seq_length; i++){
        params.timestep = i;
        for (int j = 0; j < no_layers; j++){
            params.q = reinterpret_cast<const T*>(qkv_buf[j]);
            params.k = reinterpret_cast<const T*>(qkv_buf[j]) + params.hidden_size_per_head*params.num_heads;
            params.v = reinterpret_cast<const T*>(qkv_buf[j]) + 2 * params.hidden_size_per_head*params.num_heads;
            params.k_cache = reinterpret_cast<T*>(key_cache[j]);
            params.v_cache = reinterpret_cast<T*>(value_cache[j]);
            params.out = reinterpret_cast<T*>(context_buf[j]);

            
            //if (default_launch)
            //   multihead_attention_<T, KERNEL_PARAMS_TYPE>(params, stream);
            //else
            mmha_launch<T, Dh, Dh_MAX, KERNEL_PARAMS_TYPE>(params, stream, v_vec_size, k_vec_size, threads_per_block);

            if (!cuda_check_error("kernel launch"))
                return false;

            if (res_host != nullptr) {
                cudaDeviceSynchronize();
                cudaMemcpy(res_host[j], context_buf[j], sizeof(T) * params.batch_size * params.num_heads*params.hidden_size_per_head, cudaMemcpyDeviceToHost);
                cuda_check_error("fwd answer copy");
            }
        }
    }
    return true;
}

template<typename T>
bool check_res(T** true_res_host, T** res_host, int no_layers, int length);

template<>
bool check_res(float** true_res_host, float** res_host, int no_layers, int length)
{
    for (int no_layer = 0; no_layer < no_layers; no_layer++){
        for (int i = 0; i < length; i++){
            float true_val = (true_res_host[no_layer][i]);
            float opt_val  = (res_host[no_layer][i]);
            float diff = std::abs(true_val - opt_val);
            if (diff/true_val > 0.01)
                return false;
        }
    }
    return true;
}

template<>
bool check_res(uint16_t** true_res_host, uint16_t** res_host, int no_layers, int length)
{
    for (int no_layer = 0; no_layer < no_layers; no_layer++){
        for (int i = 0; i < length; i++){
            float true_val = half_to_float(true_res_host[no_layer][i]);
            float opt_val  = half_to_float(res_host[no_layer][i]);
            float diff = std::abs(true_val - opt_val);
            if (diff/true_val > 0.01){
                return false;
            }
        }
    }
    return true;
}


template<typename T, int size_per_head>
void call_precision_headsize(){
    Masked_multihead_attention_params<T> params;
    
    memset(&params, 0, sizeof(params));
    

    int head_num = 32;
    int hidden_units = head_num * size_per_head;
    int no_layers = 16;
    params.num_heads = head_num;
    params.hidden_size_per_head = size_per_head;

    params.q_bias = nullptr;
    params.k_bias = nullptr;
    params.v_bias = nullptr;
    params.stride = 3 * hidden_units;
    params.finished = nullptr;
    params.cache_indir = nullptr;
    params.length_per_sample = nullptr;
    params.input_lengths = nullptr;
    params.beam_width = 1;
    params.rotary_embedding_dim = 0;
    params.inv_sqrt_dh = 1.F / (sqrtf((float)params.hidden_size_per_head));
    params.relative_attention_bias_stride = 0;

    std::vector<int> batch_sizes = {1};
    std::vector<int> max_tokenss = {1024};
    std::vector<int> max_input_lens = {1024};
    //std::vector<int> batch_sizes = {8};
    //std::vector<int> max_tokenss = {1024};
    //std::vector<int> max_input_lens = {1024};


    bool save_res = false;
    std::string path = "res_settings.csv";

    std::ofstream res_file;
   if (save_res){
       std::ifstream f(path);
       bool exists = f.good();
       if (!exists){
           res_file.open(path, std::ios::app);
           res_file << "batch,prompt_len,max_tokens,tpv,tpk,tpb,precision,size_per_head,time\n";
           res_file.close();
       }
    }



    for (int batch_size : batch_sizes){
        for (int max_tokens : max_tokenss){
            for (int max_input_len : max_input_lens){
                params.batch_size = batch_size;
                params.max_input_len = max_input_len;
                params.seq_length = max_input_len+max_tokens;

                /*
                int tpvs[] = {2, 4, -1};
                if (sizeof(T) == 2)
                    tpvs[2] = 8;
                else
                    tpvs[2] = 1;
                int tpks[] = {1, 2, 4};
                int tpbs[] = {64, 128, 256};
                */

                
                //int tpvs[] = {2, 4, 8};
                int tpvs[] = {4};
                int tpks[] = {1};
                int tpbs[] = {64, 128, 256};


                T** qkv_buf; T** key_cache; T** value_cache; T** context_buf; T** true_res_host; T** res_host;
                allocate_mem<T>(params, no_layers, &qkv_buf, &key_cache, &value_cache, &context_buf, &true_res_host, &res_host);
                cudaDeviceSynchronize();

                int reps=5; int warmup=3;

                for (int v = 0; v < sizeof(tpvs)/sizeof(tpvs[0]); v++){
                    for (int k = 0; k < sizeof(tpks)/sizeof(tpks[0]); k++){
                        for (int b = 0; b < sizeof(tpbs)/sizeof(tpbs[0]); b++){

                            reset_device_res<T>(params, context_buf, no_layers);
                            bool success = forward<T, size_per_head, size_per_head, Masked_multihead_attention_params<T>>(params, 0, tpvs[v], tpks[k], tpbs[b], no_layers, qkv_buf, key_cache, value_cache, context_buf, false, res_host);
                            if (!success){
                                printf("tpv=%d, tpk=%d,tpb=%d, FAILED!\n", tpvs[v], tpks[k], tpbs[b]);
                                continue;
                            }

                            for (int r = 0; r < warmup; r++)
                                forward<T, size_per_head, size_per_head, Masked_multihead_attention_params<T>>(params, 0, tpvs[v], tpks[k], tpbs[b], no_layers, qkv_buf, key_cache, value_cache, context_buf, false);
                            cudaDeviceSynchronize();

                            struct timeval start, end;
                            float secs_used, ms_took;
                            gettimeofday(&start, NULL);

                            for (int r = 0; r < reps; r++)
                                forward<T, size_per_head, size_per_head, Masked_multihead_attention_params<T>>(params, 0, tpvs[v], tpks[k], tpbs[b], no_layers, qkv_buf, key_cache, value_cache, context_buf, false);
                            cudaDeviceSynchronize();

                            gettimeofday(&end, NULL);
                            secs_used=(end.tv_sec - start.tv_sec); //avoid overflow by subtracting first
                            ms_took= (((secs_used*1000000.0) + end.tv_usec) - (start.tv_usec))/1000.0;
                            printf("batch=%d,prompt_len=%d,max_tokens=%d,tpv=%d,tpk=%d,tpb=%d,precision=%lu,size_per_head=%d,time%f\n", batch_size, max_input_len, max_tokens, tpvs[v], tpks[k], tpbs[b], sizeof(T)*8,size_per_head, ms_took/(float)reps);

                            if (save_res){
                               res_file.open(path, std::ios::app);
                               res_file << batch_size<<"," << max_input_len<<","<<max_tokens<<","<<tpvs[v]<<","<<tpks[k]<<","<<tpbs[b]<<","<<sizeof(T)*8<<","<<size_per_head<<","<<ms_took/(float)reps<<"\n";
                               res_file.close();
                            }
                        }
                    }
                }

                free_mem<T>(no_layers, qkv_buf, key_cache, value_cache, context_buf, true_res_host, res_host);
            }
        }
    }


}


int main(int argc, char* argv[]) {
    //using T = uint16_t;
    //using T = float;
    //using T = typename TypeConverter<float2>::Type;

    //using DataType = typename T;//typename TypeConverter<T>::Type;
    std::vector<int> precisions = {32};
    std::vector<int> size_per_heads = {64};
    
    for (int precision : precisions){
        for (int size_per_head : size_per_heads){
            switch (precision){
                case 16:
                    /*
                    switch (size_per_head){
                        case 64:
                            call_precision_headsize<uint16_t, 64>();
                            break;
                        case 128:
                            call_precision_headsize<uint16_t, 128>();
                            break;
                        default:
                            printf("Not support head size\n");
                            break;
                            
                    }
                    */
                    break;
                case 32:
                    switch (size_per_head){
                        case 64:
                            call_precision_headsize<float, 64 >();
                            break;
                        case 128:
                            call_precision_headsize<float, 128>();
                            break;
                        default:
                            printf("Not support head size\n");
                            break;
                            
                    }
            }
        }
    }

    return 0;
}
