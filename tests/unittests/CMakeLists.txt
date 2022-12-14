# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_executable(test_gemm test_gemm.cu)
target_link_libraries(test_gemm PUBLIC -lcublas -lcudart -lcurand gemm cublasMMWrapper)

add_executable(find_setting find_settings.cu)
target_link_libraries(find_setting PUBLIC -lcublas -lcudart -lcurand gemm cublasMMWrapper)

add_executable(test_cost_model test_cost_model.cu)
target_link_libraries(test_cost_model PUBLIC -lcublas -lcudart -lcurand gemm cublasMMWrapper unfused_attention_kernels)

add_executable(run_attention_kernel run_attention_kernel.cu)
target_link_libraries(run_attention_kernel PUBLIC -lcublas -lcudart -lcurand gemm cublasMMWrapper)

add_executable(test_sampling test_sampling.cu)
target_link_libraries(test_sampling PUBLIC
                      -lcublas -lcublasLt -lcudart
                      cublasMMWrapper memory_utils
                      DynamicDecodeLayer TopKSamplingLayer TopPSamplingLayer TopKTopPSamplingLayer)

add_executable(test_logprob_kernels test_logprob_kernels.cu)
target_link_libraries(test_logprob_kernels PUBLIC
                    -lcublas -lcublasLt -lcudart
                    logprob_kernels memory_utils)
