# Makefile for the MoE token-packing library.
#
# Layout (see README):
#   include/moe/      public headers
#   src/              strategy implementations (compiled to .o)
#   apps/             binaries (one .cu per binary)
#   build/            intermediate objects
#
# Usage:
#   make              # build everything currently wired up
#   make objects      # just the .o files
#   make clean
#
# CUDA arch / cuBLAS / NCCL path are overridable from the command line:
#   make ARCH=sm_89
#   make CUBLAS12=/alt/path
#   make NCCL=/alt/path

# ---- toolchain -------------------------------------------------------------
NVCC      ?= nvcc
ARCH      ?= sm_86
CUBLAS12  ?= $(HOME)/miniconda3/envs/myenv/lib/python3.9/site-packages/nvidia/cublas
CUTLASS   ?= third_party/cutlass
NCCL      ?= $(HOME)/.local/lib/python3.6/site-packages/nvidia/nccl

INCLUDES  := -Iinclude -I$(CUBLAS12)/include
CUTLASS_INC := -I$(CUTLASS)/include -I$(CUTLASS)/tools/util/include
NCCL_INC  := -I$(NCCL)/include
NVCC_FLAGS_EXTRA ?=
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) --expt-relaxed-constexpr $(INCLUDES) $(NVCC_FLAGS_EXTRA)

# cuBLAS 12 from conda (driver 572.83 is incompatible with CUDA 11.6 cuBLAS).
CUBLAS_LINK := -Xlinker -rpath=$(CUBLAS12)/lib \
               -Xlinker $(CUBLAS12)/lib/libcublas.so.12 \
               -Xlinker $(CUBLAS12)/lib/libcublasLt.so.12
NVTX_LINK   := -lnvToolsExt
NCCL_LINK   := -Xlinker -rpath=$(NCCL)/lib \
               -Xlinker $(NCCL)/lib/libnccl.so.2

# ---- directories -----------------------------------------------------------
BUILD_DIR := build
OBJ_DIR   := $(BUILD_DIR)/obj
BIN_DIR   := $(BUILD_DIR)/bin

$(shell mkdir -p $(OBJ_DIR) $(BIN_DIR))

# ---- source sets -----------------------------------------------------------
# Common host-side objects (.cpp).
COMMON_SRCS := src/common/npy_io.cpp src/common/cpu_reference.cpp
COMMON_OBJS := $(patsubst src/%.cpp,$(OBJ_DIR)/%.o,$(COMMON_SRCS))

# Scatter strategy objects + factory. Factory lives here so every binary that
# links strategies also gets dispatch.
SCATTER_SRCS := src/scatter/factory.cu \
                src/scatter/scatter_atomic.cu \
                src/scatter/scatter_sort.cu \
                src/scatter/scatter_warp.cu \
                src/scatter/scatter_vec.cu \
                src/scatter/scatter_csort.cu
SCATTER_OBJS := $(patsubst src/%.cu,$(OBJ_DIR)/%.o,$(SCATTER_SRCS))

# Apps (each compiles + links into its own binary).
APP_SRCS := apps/verify.cu apps/bench_scatter.cu apps/bench_e2e.cu \
            apps/grouped_gemm_test.cu apps/bench_ep.cu apps/bench_prefill.cu \
            apps/bench_blocksparse.cu
APP_BINS := $(patsubst apps/%.cu,$(BIN_DIR)/%,$(APP_SRCS))

# ---- default goal ----------------------------------------------------------
.PHONY: all objects clean verify
all: objects $(APP_BINS)

objects: $(COMMON_OBJS) $(SCATTER_OBJS)

verify: $(BIN_DIR)/verify

# ---- pattern rules ---------------------------------------------------------
$(OBJ_DIR)/scatter/%.o: src/scatter/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/common/%.o: src/common/%.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Per-app link-extras & extra includes. Defaults empty; override per-target.
LINK_EXTRA ?=
EXTRA_INC  ?=
$(BIN_DIR)/bench_e2e:          LINK_EXTRA := $(CUBLAS_LINK) $(NVTX_LINK)
$(BIN_DIR)/grouped_gemm_test:  LINK_EXTRA := $(CUBLAS_LINK)
$(BIN_DIR)/grouped_gemm_test:  EXTRA_INC  := $(CUTLASS_INC)
$(BIN_DIR)/bench_ep:           LINK_EXTRA := $(NCCL_LINK)
$(BIN_DIR)/bench_ep:           EXTRA_INC  := $(NCCL_INC)
$(BIN_DIR)/bench_prefill:      LINK_EXTRA := $(CUBLAS_LINK) $(NCCL_LINK) $(NVTX_LINK)
$(BIN_DIR)/bench_prefill:      EXTRA_INC  := $(NCCL_INC)
$(BIN_DIR)/bench_blocksparse:  LINK_EXTRA := $(CUBLAS_LINK)

$(BIN_DIR)/%: apps/%.cu $(SCATTER_OBJS) $(COMMON_OBJS)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_INC) $< $(SCATTER_OBJS) $(COMMON_OBJS) $(LINK_EXTRA) -o $@

# ---- housekeeping ----------------------------------------------------------
clean:
	rm -rf $(BUILD_DIR)
