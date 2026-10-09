// Copyright 2026 Qwen38 native contributors. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>

// All pointers/graph handles belong to this process. Caller retains device
// buffers, CUDA graphs, their memory pools, and kernel modules until close.
// A pipe has one in-flight gather. Its store must outlive it. A plan borrows
// pipes and graph executables; it NEVER frees or updates a borrowed graph.
extern "C" {
const char* q38_native_last_error();
void* q38_pipe_create(void* store, size_t row_bytes, size_t max_rows, int device);
int64_t q38_pipe_issue(void* pipe, const int64_t* device_ids, size_t count,
                       uintptr_t stream);
int q38_pipe_collect(void* pipe, int64_t ticket, void* bf16_output, size_t count,
                     uintptr_t stream);
int q38_pipe_pending(void* pipe);
int q38_pipe_close(void* pipe);
void* q38_plan_create(int device);
int q38_plan_add_graph(void* plan, uintptr_t graph_exec);
int q38_plan_add_issue(void* plan, void* pipe, const int64_t* ids, size_t count);
int q38_plan_add_collect(void* plan, void* pipe, void* output, size_t count);
int q38_plan_seal(void* plan);
int q38_plan_replay(void* plan, uintptr_t stream);
int q38_plan_close(void* plan);
int q38_decode_fp8(const uint8_t* input, void* output, size_t count, uintptr_t stream);
}
