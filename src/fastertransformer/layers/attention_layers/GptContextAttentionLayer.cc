/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/layers/attention_layers/GptContextAttentionLayer.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"

namespace fastertransformer {

template<typename T>
void GptContextAttentionLayer<T>::forward(std::vector<fastertransformer::Tensor>* output_tensors,
                                          const std::vector<fastertransformer::Tensor>* input_tensors,
                                          const AttentionWeight<T>* attention_weights)
{
    // input_tensors:
    //      input_query [batch_size * seq_len, hidden_dimension]
    //      attention_mask [batch_size, 1, seq_len, seq_len]
    //      is_final_layer [1], bool on cpu

    // output_tensors:
    //      attention_out [batch_size * seq_len, hidden_dimension]
    //      key_cache [batch, local_head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [batch, local_head_num, max_seq_len, size_per_head]

    FT_CHECK(input_tensors->size() == 3);
    FT_CHECK(output_tensors->size() == 3);
    FT_CHECK(output_tensors->at(1).shape.size() == 5);
    FT_CHECK(output_tensors->at(2).shape.size() == 4 || output_tensors->at(2).shape.size() == 3);
    // FT_CHECK(isValidBatchSize(input_tensors->at(1).shape[0]));
    // FT_CHECK(isValidSeqLen(input_tensors->at(1).shape[2]));
    const int request_batch_size = input_tensors->at(1).shape[0];
    const int request_seq_len = input_tensors->at(1).shape[2];
    allocateBuffer(request_batch_size, request_seq_len);
    sync_check_cuda_error();

    T* attention_out = (T*)output_tensors->at(0).data;
    const T* attention_input = (const T*)input_tensors->at(0).data;
    const T* attention_mask = (const T*)input_tensors->at(1).data;
    const bool is_final = *((bool*)(input_tensors->at(2).data));

    const int m = input_tensors->at(0).shape[0];

#ifdef SPARSITY_ENABLED
    const int m_padded = 8 * div_up(m, 8);
    if (sparse_ && cublas_wrapper_->isUseSparse(1, 3 * local_hidden_units_, m_padded, hidden_units_)) {
        cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                3 * local_hidden_units_,
                                m_padded,
                                hidden_units_,
                                attention_weights->query_weight.sp_kernel,
                                attention_input,
                                qkv_buf_);
    }
    else {
#endif
        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              3 * local_hidden_units_,  // n
                              m,
                              hidden_units_,  // k
                              attention_weights->query_weight.kernel,
                              3 * local_hidden_units_,  // n
                              attention_input,
                              hidden_units_,  // k
                              qkv_buf_,
                              3 * local_hidden_units_ /* n */);
#ifdef SPARSITY_ENABLED
    }
#endif
    sync_check_cuda_error();
    invokeAddFusedQKVBiasTranspose(q_buf_2_,
                                   k_buf_2_,
                                   v_buf_2_,
                                   qkv_buf_,
                                   attention_weights->query_weight.bias,
                                   request_batch_size,
                                   request_seq_len,
                                   local_head_num_,
                                   size_per_head_,
                                   rotary_embedding_dim_,
                                   stream_);
    sync_check_cuda_error();

    const int max_seq_len = (int)(output_tensors->at(1).shape[3]);
    // Use batch major
    // put k/v_buf from shape [B, H, L, Dh]
    // to cache [B, H, Dh/x, L, x]  and [B, H, L, Dh/x, x]
    invokeTranspose4dBatchMajor((T*)output_tensors->at(1).data,
                                (T*)output_tensors->at(2).data,
                                k_buf_2_,
                                v_buf_2_,
                                request_batch_size,
                                request_seq_len,
                                max_seq_len,
                                size_per_head_,
                                local_head_num_,
                                stream_);
    sync_check_cuda_error();

    if (is_final == false) {
        T* kernel_out = qkv_buf_3_;
        bool with_context_attn; std::stringstream ss(std::getenv("with_context_attn")); ss >> with_context_attn;
        if (with_context_attn){
            bool flash_var; std::stringstream ss(std::getenv("context_attn_flash")); ss >> flash_var;
            bool faster_var; std::stringstream ss2(std::getenv("context_attn_faster")); ss2 >> faster_var;

            bool validate_and_faster = faster_var && flash_var;
            bool validate_and_flash = !faster_var && !flash_var;
            bool validate = validate_and_faster || validate_and_flash;
            
            bool run_flash =flash_var || validate_and_flash;
            bool run_faster = faster_var || validate_and_flash;
            bool use_flash = !faster_var;

            if (run_flash){
                bool is_bf16 = false;

                mha_faster_fwd(qkv_buf_, 
                    flash_buf,
                    flash_otemp,
                    flash_softmax_lse,
                    flash_seq_lens,
                    request_batch_size,
                    request_seq_len,
                    size_per_head_,
                    local_head_num_,
                    is_bf16,
                    stream_
                    );

                sync_check_cuda_error();

            }

            if (run_faster){
                const cudaDataType_t gemm_data_type = getCudaDataType<T>();
                if (is_qk_buf_float_ == true && gemm_data_type != CUDA_R_32F) {
                    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                                        CUBLAS_OP_N,
                                                        request_seq_len,
                                                        request_seq_len,
                                                        size_per_head_,
                                                        1.0f,
                                                        k_buf_2_,
                                                        gemm_data_type,
                                                        size_per_head_,
                                                        request_seq_len * size_per_head_,
                                                        q_buf_2_,
                                                        gemm_data_type,
                                                        size_per_head_,
                                                        request_seq_len * size_per_head_,
                                                        0.0f,
                                                        qk_buf_float_,
                                                        CUDA_R_32F,
                                                        request_seq_len,
                                                        request_seq_len * request_seq_len,
                                                        request_batch_size * local_head_num_,
                                                        CUDA_R_32F);
                    sync_check_cuda_error();
                    T scalar = 1 / sqrtf(size_per_head_ * 1.0f);
                    invokeMaskedSoftMax(qk_buf_,
                                        qk_buf_float_,
                                        attention_mask,
                                        request_batch_size,
                                        request_seq_len,
                                        local_head_num_,
                                        scalar,
                                        stream_);
                    sync_check_cuda_error();
                }
                else {
                    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                                        CUBLAS_OP_N,
                                                        request_seq_len,
                                                        request_seq_len,
                                                        size_per_head_,
                                                        k_buf_2_,
                                                        size_per_head_,
                                                        request_seq_len * size_per_head_,
                                                        q_buf_2_,
                                                        size_per_head_,
                                                        request_seq_len * size_per_head_,
                                                        qk_buf_,
                                                        request_seq_len,
                                                        request_seq_len * request_seq_len,
                                                        request_batch_size * local_head_num_);

                    T scalar = 1 / sqrtf(size_per_head_ * 1.0f);
                    invokeMaskedSoftMax(qk_buf_,
                                        qk_buf_,
                                        attention_mask,
                                        request_batch_size,
                                        request_seq_len,
                                        local_head_num_,
                                        scalar,
                                        stream_);
                    sync_check_cuda_error();
                }

                cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                                    CUBLAS_OP_N,
                                                    size_per_head_,
                                                    request_seq_len,
                                                    request_seq_len,
                                                    v_buf_2_,
                                                    size_per_head_,
                                                    request_seq_len * size_per_head_,
                                                    qk_buf_,
                                                    request_seq_len,
                                                    request_seq_len * request_seq_len,
                                                    qkv_buf_2_,
                                                    size_per_head_,
                                                    request_seq_len * size_per_head_,
                                                    request_batch_size * local_head_num_);


                invokeTransposeQKV(
                    qkv_buf_3_, qkv_buf_2_, request_batch_size, request_seq_len, local_head_num_, size_per_head_, stream_);
                sync_check_cuda_error();
            }

            if (validate){
                T* flash_attn_out = (T*)malloc(sizeof(T) * request_batch_size * request_seq_len * local_hidden_units_);
                cudaMemcpy(flash_attn_out, flash_buf, sizeof(T) * request_batch_size * request_seq_len * local_hidden_units_, cudaMemcpyDeviceToHost);  

                T* faster_attn_out = (T*)malloc(sizeof(T) * request_batch_size * request_seq_len * local_hidden_units_);
                cudaMemcpy(faster_attn_out, qkv_buf_3_, sizeof(T) * request_batch_size * request_seq_len * local_hidden_units_, cudaMemcpyDeviceToHost);  

                printf("ATTN DIFF=======================================================================\n");
                printf("request_batch_size=%d, request_seq_len=%d , local_hidden_units_=%d\n",request_batch_size, request_seq_len , local_hidden_units_);
                for (int i = 0; i < request_batch_size * request_seq_len * local_hidden_units_; i++)
                {
                    printf("faster=%f, flash=%f, diff=%f\n", (float)faster_attn_out[i], (float)flash_attn_out[i], std::abs((float)(faster_attn_out[i]-flash_attn_out[i])));
                }
                free(faster_attn_out);
                free(flash_attn_out);
            }

            if (use_flash){
                kernel_out = flash_buf;
            }
        }

#ifdef SPARSITY_ENABLED
        if (sparse_ && cublas_wrapper_->isUseSparse(1, hidden_units_, m_padded, local_hidden_units_)) {
            cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    hidden_units_,
                                    m_padded,
                                    local_hidden_units_,
                                    attention_weights->attention_output_weight.sp_kernel,
                                    kernel_out,
                                    attention_out);
        }
        else {
#endif
            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  hidden_units_,
                                  m,
                                  local_hidden_units_,
                                  attention_weights->attention_output_weight.kernel,
                                  hidden_units_,
                                  kernel_out,
                                  local_hidden_units_,
                                  attention_out,
                                  hidden_units_);
#ifdef SPARSITY_ENABLED
        }
#endif
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template<typename T>
GptContextAttentionLayer<T>::GptContextAttentionLayer(size_t max_batch_size,
                                                      size_t max_seq_len,
                                                      size_t head_num,
                                                      size_t size_per_head,
                                                      cudaStream_t stream,
                                                      cublasMMWrapper* cublas_wrapper,
                                                      IAllocator* allocator,
                                                      bool is_free_buffer_after_forward,
                                                      bool is_qk_buf_float,
                                                      bool sparse):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    local_head_num_(head_num),
    local_hidden_units_(local_head_num_ * size_per_head),
    rotary_embedding_dim_(0),
    is_qk_buf_float_(is_qk_buf_float)
{
}

template<typename T>
GptContextAttentionLayer<T>::GptContextAttentionLayer(size_t max_batch_size,
                                                      size_t max_seq_len,
                                                      size_t head_num,
                                                      size_t size_per_head,
                                                      size_t local_head_num,
                                                      cudaStream_t stream,
                                                      cublasMMWrapper* cublas_wrapper,
                                                      IAllocator* allocator,
                                                      bool is_free_buffer_after_forward,
                                                      bool is_qk_buf_float,
                                                      bool sparse):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    local_head_num_(local_head_num),
    local_hidden_units_(local_head_num_ * size_per_head),
    rotary_embedding_dim_(0),
    is_qk_buf_float_(is_qk_buf_float)
{
}

template<typename T>
GptContextAttentionLayer<T>::GptContextAttentionLayer(size_t max_batch_size,
                                                      size_t max_seq_len,
                                                      size_t head_num,
                                                      size_t size_per_head,
                                                      size_t local_head_num,
                                                      size_t rotary_embedding_dim,
                                                      cudaStream_t stream,
                                                      cublasMMWrapper* cublas_wrapper,
                                                      IAllocator* allocator,
                                                      bool is_free_buffer_after_forward,
                                                      bool is_qk_buf_float,
                                                      bool sparse):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    local_head_num_(local_head_num),
    local_hidden_units_(local_head_num_ * size_per_head),
    rotary_embedding_dim_(rotary_embedding_dim),
    is_qk_buf_float_(is_qk_buf_float)
{
}

template<typename T>
GptContextAttentionLayer<T>::GptContextAttentionLayer(GptContextAttentionLayer<T> const& attention_layer):
    BaseAttentionLayer<T>(attention_layer.stream_,
                          attention_layer.cublas_wrapper_,
                          attention_layer.allocator_,
                          attention_layer.is_free_buffer_after_forward_,
                          attention_layer.sparse_),
    max_batch_size_(attention_layer.max_batch_size_),
    max_seq_len_(attention_layer.max_seq_len_),
    head_num_(attention_layer.head_num_),
    size_per_head_(attention_layer.size_per_head_),
    hidden_units_(attention_layer.hidden_units_),
    local_head_num_(attention_layer.local_head_num_),
    local_hidden_units_(attention_layer.local_hidden_units_),
    rotary_embedding_dim_(attention_layer.rotary_embedding_dim_),
    is_qk_buf_float_(attention_layer.is_qk_buf_float_)
{
}

template<typename T>
GptContextAttentionLayer<T>::~GptContextAttentionLayer()
{
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void GptContextAttentionLayer<T>::allocateBuffer()
{
    FT_CHECK(false);
    if (is_allocate_buffer_ == false) {
        qkv_buf_ = (T*)allocator_->malloc(sizeof(T) * 3 * max_batch_size_ * max_seq_len_ * local_hidden_units_, true);
        q_buf_2_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * local_hidden_units_, true);
        k_buf_2_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * local_hidden_units_, true);
        v_buf_2_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * local_hidden_units_, true);

        qk_buf_ =
            (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * local_head_num_ * max_seq_len_ * max_seq_len_, true);
        qkv_buf_2_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * local_hidden_units_, true);
        qkv_buf_3_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * local_hidden_units_, true);

        bool flash_var; std::stringstream ss(std::getenv("context_attn_flash")); ss >> flash_var;
        bool faster_var; std::stringstream ss2(std::getenv("context_attn_faster")); ss2 >> faster_var;
        bool no_flash = faster_var && !flash_var;
        if (!no_flash){
            flash_buf = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * local_hidden_units_, true);
            flash_otemp = (float*)allocator_->malloc(sizeof(float) * max_batch_size_ * max_seq_len_ * local_hidden_units_, true);
            flash_softmax_lse = (float*)allocator_->malloc(sizeof(float) * max_batch_size_ * ((max_seq_len_ + 16 - 1) / 16) * 16 * local_head_num_, true);
            flash_seq_lens = (int*)allocator_->malloc(sizeof(int) * (max_batch_size_+1), true);

            int* seq_lens = (int*)malloc((max_batch_size_+1)*sizeof(int));
            for (int i = 0; i <= max_batch_size_; i++) 
                seq_lens[i] = (i)*max_seq_len_;
            cudaMemcpy(flash_seq_lens, seq_lens, (max_batch_size_+1)*sizeof(int), cudaMemcpyHostToDevice); 
            free(seq_lens);
        }

        if (is_qk_buf_float_ == true) {
            qk_buf_float_ = (float*)allocator_->malloc(
                sizeof(float) * max_batch_size_ * local_head_num_ * max_seq_len_ * max_seq_len_, true);
        }

        is_allocate_buffer_ = true;
    }
}

template<typename T>
void GptContextAttentionLayer<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    qkv_buf_ = (T*)allocator_->reMalloc(qkv_buf_, sizeof(T) * 3 * batch_size * seq_len * local_hidden_units_, true);
    q_buf_2_ = (T*)allocator_->reMalloc(q_buf_2_, sizeof(T) * batch_size * seq_len * local_hidden_units_, true);
    k_buf_2_ = (T*)allocator_->reMalloc(k_buf_2_, sizeof(T) * batch_size * seq_len * local_hidden_units_, true);
    v_buf_2_ = (T*)allocator_->reMalloc(v_buf_2_, sizeof(T) * batch_size * seq_len * local_hidden_units_, true);

    qk_buf_ = (T*)allocator_->reMalloc(qk_buf_, sizeof(T) * batch_size * local_head_num_ * seq_len * seq_len, true);
    qkv_buf_2_ = (T*)allocator_->reMalloc(qkv_buf_2_, sizeof(T) * batch_size * seq_len * local_hidden_units_, true);
    qkv_buf_3_ = (T*)allocator_->reMalloc(qkv_buf_3_, sizeof(T) * batch_size * seq_len * local_hidden_units_, true);

    bool flash_var; std::stringstream ss(std::getenv("context_attn_flash")); ss >> flash_var;
    bool faster_var; std::stringstream ss2(std::getenv("context_attn_faster")); ss2 >> faster_var;
    bool no_flash = faster_var && !flash_var;
    if (!no_flash){
        flash_buf = (T*)allocator_->reMalloc(flash_buf, sizeof(T) * batch_size * seq_len * local_hidden_units_, true);
        flash_otemp = (float*)allocator_->reMalloc(flash_otemp, sizeof(float) * batch_size * seq_len * local_hidden_units_, true);
        flash_softmax_lse = (float*)allocator_->reMalloc(flash_softmax_lse, sizeof(float) * batch_size * ((seq_len + 16 - 1) / 16) * 16 * local_head_num_, true);
        flash_seq_lens = (int*)allocator_->reMalloc(flash_seq_lens, sizeof(int) * (batch_size+1), true);

        int* seq_lens = (int*)malloc((batch_size+1)*sizeof(int));
        for (int i = 0; i <= batch_size; i++) 
            seq_lens[i] = (i)*seq_len;
        cudaMemcpy(flash_seq_lens, seq_lens, (batch_size+1)*sizeof(int), cudaMemcpyHostToDevice); 
        free(seq_lens);
    }

    if (is_qk_buf_float_ == true) {
        qk_buf_float_ = (float*)allocator_->reMalloc(
            qk_buf_float_, sizeof(float) * batch_size * local_head_num_ * seq_len * seq_len, true);
    }
    is_allocate_buffer_ = true;
}

template<typename T>
void GptContextAttentionLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        allocator_->free(qkv_buf_);
        allocator_->free(q_buf_2_);
        allocator_->free(k_buf_2_);
        allocator_->free(v_buf_2_);
        allocator_->free(qk_buf_);
        allocator_->free(qkv_buf_2_);
        allocator_->free(qkv_buf_3_);

        bool flash_var; std::stringstream ss(std::getenv("context_attn_flash")); ss >> flash_var;
        bool faster_var; std::stringstream ss2(std::getenv("context_attn_faster")); ss2 >> faster_var;
        bool no_flash = faster_var && !flash_var;
        if (!no_flash){
            allocator_->free(flash_buf);
            allocator_->free(flash_otemp);
            allocator_->free(flash_softmax_lse);
            allocator_->free(flash_seq_lens);
        }

        if (is_qk_buf_float_ == true) {
            allocator_->free(qk_buf_float_);
        }
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool GptContextAttentionLayer<T>::isValidBatchSize(size_t batch_size)
{
    if (batch_size <= max_batch_size_) {
        return true;
    }
    else {
        freeBuffer();
        max_batch_size_ = batch_size * 1.2;
        return true;
    }
}

template<typename T>
bool GptContextAttentionLayer<T>::isValidSeqLen(size_t seq_len)
{
    if (seq_len <= max_seq_len_) {
        return true;
    }
    else {
        freeBuffer();
        max_seq_len_ = seq_len * 1.2;
        return true;
    }
}

template class GptContextAttentionLayer<float>;
template class GptContextAttentionLayer<half>;
#ifdef ENABLE_BF16
template class GptContextAttentionLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
