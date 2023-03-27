// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

void SequencePoolingCuda(
  cudaStream_t stream,
  const int batch_size,
  const int hidden_size,
  const int num_sequences,
  const int sequence_length_for_split,
  const float* input,
  const int64_t* sentence_lengthes,
  float* output,
  int64_t* inclusive_len_prefix_sum);

void SequencePoolingCuda(
  cudaStream_t stream,
  const int batch_size,
  const int hidden_size,
  const int num_sequences,
  const int sequence_length_for_split,
  const half* input,
  const int64_t* sentence_lengthes,
  half* output,
  int64_t* inclusive_len_prefix_sum);

void SequencePoolingCPU(
  const int batch_size,
  const int hidden_size,
  const int num_sequences,
  const int sequence_length_for_split,
  const float* input,
  const int64_t* sentence_lengthes,
  float* output,
  int64_t* inclusive_len_prefix_sum);
