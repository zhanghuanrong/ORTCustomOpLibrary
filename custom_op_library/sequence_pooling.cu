// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <cstdint>
#include <iostream>

#include "sequence_pooling.h"

using namespace std;

#define CUDA_CHECK(apiFuncCall)                                               \
  do                                                                          \
  {                                                                           \
    cudaError_t _status = apiFuncCall;                                        \
    if (_status != cudaSuccess)                                               \
    {                                                                         \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",    \
              __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status)); \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

// An example
// In: input: [1, 4096, 768]
// In: sen_lens: [1, 47]     contains like [30, 40, 20, ....., 96] and sum up to 4096
// Out: output: [1, 256, 768]
//      where [0, 0:46, 768] is the max pooling result of input along axis=1 by sen_lens
//      and [0, 47:256, 768] part is all zeros

template <typename T, int NumOutputSentences>
__global__ void
SequencePoolingWithPrefixSumedCudaKernel(const T *input,
                                         const int64_t *inclusive_prefix_sum,
                                         const int passage_length,
                                         const int num_sentences,
                                         T *output,
                                         const int hidden_size)
{
  const int sentence_id = blockIdx.y;
  const int batch_id = blockIdx.z;
  const int hidden_id = blockDim.x * blockIdx.x + threadIdx.x;

  if (hidden_id >= hidden_size)
    return;

  // move to batch start for the input/output and inclusive_prefix_sum
  input += (batch_id * passage_length * hidden_size);
  inclusive_prefix_sum += (batch_id * num_sentences);
  output += (batch_id * NumOutputSentences * hidden_size);

  const int output_offset = sentence_id * hidden_size + hidden_id;
  if (sentence_id >= num_sentences)
  {
    output[output_offset] = T{0};
    return;
  }

  const int start_token = ((sentence_id > 0) ? (int)(inclusive_prefix_sum[sentence_id - 1]) : 0);
  const auto ps_this = inclusive_prefix_sum[sentence_id];
  const auto ps_next = (sentence_id < num_sentences - 1) ? inclusive_prefix_sum[sentence_id + 1] : ps_this;
  int sentence_length = (int)(ps_this)-start_token;
  if (ps_next == ps_this)
    sentence_length = 0;

  // move input to it start token's hidden element
  input += (start_token * hidden_size + hidden_id);

  T local_max = ((sentence_length > 0) ? (*input) : T{0});
  for (int i = 1; i < sentence_length; ++i)
  {
    input += hidden_size;
    const T value = *input;
    if ((float)local_max < (float)value) {
      local_max = value;
    }
  }
  output[output_offset] = local_max;
}

__global__ void InclusivePrefixSum(const int64_t *sen_lengthes, int num_sen, int64_t *ps_len)
{
  typedef cub::BlockScan<int64_t, 256> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  const int offset = blockIdx.x * num_sen + threadIdx.x;
  int64_t thread_data = (threadIdx.x < num_sen) ? (sen_lengthes[offset]) : 0LL;
  BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);

  if (threadIdx.x < num_sen)
  {
    ps_len[offset] = thread_data;
  }
}

constexpr int NumOutputSentences = 256;

void SequencePoolingCuda(cudaStream_t stream,
                         const int batch_size,
                         const int hidden_size,
                         const int num_sentences,
                         const int passage_length,
                         const float *input,
                         const int64_t *sentence_lengthes,
                         float *output,
                         int64_t *inclusive_prefix_sum)
{
  InclusivePrefixSum<<<batch_size, NumOutputSentences, 0, stream>>>(
    sentence_lengthes, num_sentences, inclusive_prefix_sum);
  CUDA_CHECK(cudaGetLastError());

  const dim3 grid((hidden_size + 256 - 1) / 256, NumOutputSentences, batch_size);
  const dim3 block(256, 1, 1);
  SequencePoolingWithPrefixSumedCudaKernel<float, NumOutputSentences><<<grid, block, 0, stream>>>(
      input, inclusive_prefix_sum, passage_length, num_sentences, output, hidden_size);
  CUDA_CHECK(cudaGetLastError());
}


void SequencePoolingCuda(cudaStream_t stream,
                         const int batch_size,
                         const int hidden_size,
                         const int num_sentences,
                         const int passage_length,
                         const half *input,
                         const int64_t *sentence_lengthes,
                         half *output,
                         int64_t *inclusive_prefix_sum)
{
  InclusivePrefixSum<<<batch_size, NumOutputSentences, 0, stream>>>(
    sentence_lengthes, num_sentences, inclusive_prefix_sum);
  CUDA_CHECK(cudaGetLastError());

  const dim3 grid((hidden_size + 256 - 1) / 256, NumOutputSentences, batch_size);
  const dim3 block(256, 1, 1);
  SequencePoolingWithPrefixSumedCudaKernel<half, NumOutputSentences><<<grid, block, 0, stream>>>(
      input, inclusive_prefix_sum, passage_length, num_sentences, output, hidden_size);
  CUDA_CHECK(cudaGetLastError());
}
