/*
    本头文件实现了根据计算能力和核函数计算最佳线程网格大小配置。
*/

#ifndef LAUNCH_H
#define LAUNCH_H

#include <iostream>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <cuda_runtime.h>
#include "check.h"

// 最大设备数量，暂定200，如果太小，请手动调整
constexpr unsigned int MAX_DEVICES = 200;

// 线程块大小，选择256，对于目前的GPU，128也是一个不错的选择，但不能选择64，
// 否则一个sm上需要驻留的线程块数可能超出限制。仅用于非启发式配置
constexpr int BLOCK_SIZE = 256;

/*
    计算最佳线程数量配置，适用于动态分配的共享内存为常数的情况。
    grid_size - 返回线程块数
    block_size - 返回块内线程数量
    kernel - 核函数
    N - 数据量的大小，如果数据量不够多不能使所有SM都能分配到至少一个线程块，则使用N来计算块数
    dynamicSMemSize - 每个线程块需要的共享内存字节数，不包括核函数内静态分配的共享内存
    block_size_limit - 每个线程块的最大线程数量，0表示硬件限制
*/
template<typename KernelFunction>
void LaunchConfig(int *grid_size,
                  int *block_size,
                  KernelFunction kernel,
                  int N = 0,
                  size_t dynamic_smem_size = 0,
                  int block_size_limit = 0) {
    OccupancyMaxPotentialBlockSize(
            grid_size, block_size, kernel, dynamic_smem_size, block_size_limit);
    if (N > 0)
        *grid_size = std::min(*grid_size, (N + *block_size - 1) / *block_size);
}

/*
    计算最佳线程数量配置，适用于动态分配的共享内存和block_size相关的情况。
    grid_size - 返回线程块数
    block_size - 返回块内线程数量
    kernel - 核函数
    BlockSizeToSMem - 一元函数，以block_size为实参调用则返回每个线程块需要的
            共享内存字节数，不包括核函数内静态分配的共享内存
    N - 数据量的大小，如果数据量不够多不能使所有SM都能分配到至少一个线程块，则使用N来计算块数
    block_size_limit - 每个线程块的最大线程数量，0表示硬件限制
*/
template<typename KernelFunction, typename UnaryFunction>
void LaunchConfig(int *grid_size,
                  int *block_size,
                  KernelFunction kernel,
                  UnaryFunction BlockSizeToSMem,
                  int N = 0,
                  int block_size_limit = 0) {
    OccupancyMaxPotentialBlockSizeVariableSMem(
            grid_size, block_size, kernel, BlockSizeToSMem, block_size_limit);
    if (N > 0)
        *grid_size = std::min(*grid_size, (N + *block_size - 1) / *block_size);
}

/*
    计算最佳线程数量配置，不使用启发式策略，直接选择计算能力所限制的最大线程块和此线程块配置下
    整个GPU所能容纳的线程块数量作为grid_size。
    此函数适用于核函数使用资源比较少的情况，如3.5的计算能力下，一个gpu有13个SM，一个SM有64K
    个32位寄存器和48KB共享内存，一个线程最多可以分配255个寄存器，一个线程块最多可以分配48KB
    共享内存，一个线程块内最多可以有1024个线程，一个SM上最多可以常驻2048个线程即2个线程块，
    所以如果每个线程的寄存器个数不超过32个，每个线程块共享内存不超过24KB，线程网格分配为26*1024
    可以达到最优性能，因为此时所有SM都已满载，而SM内部所有线程都可常驻且无数据保存至全局内存，
    并且此时无调度开销，因此性能已达最优。(满载是指SM上容纳的线程数量已经达到最大值，此时访存
    延迟可以很好地被隐藏起来)
    注意：计算能力3.0以下的设备上，一个SM上最多可常驻的线程数量不是线程块中最大线程数量的整数倍，
    使用此函数不能得到最优结果。截止目前为止(2019年10月)，所有计算能力大于3.0的设备都可以使用。
    提示：可以使用nvcc --ptxas-options=-v main.cu来查看其中的核函数的资源需求量。
*/
inline void LaunchConfigMax(int *grid_size, int *block_size) {
    // 记录某个设备计算出来的线程网格配置，grids[device_id].first为线程块大小，
    // grids[device_id].second为线程块数量
    static std::pair<int, int> grids[MAX_DEVICES];

    int device_id = 0;
    cudaGetDevice(&device_id);
    if (device_id >= MAX_DEVICES)
        throw std::runtime_error{"设备数量过大，请调整最大设备数量"};

    if (grids[device_id].first == 0) {
        int max_threads_SM = 0;
        int max_threads_block = 0;
        int num_SM = 0;

        CHECK(cudaDeviceGetAttribute(
                &max_threads_SM, cudaDevAttrMaxThreadsPerMultiProcessor, device_id));
        CHECK(cudaDeviceGetAttribute(
                &max_threads_block, cudaDevAttrMaxThreadsPerBlock, device_id));
        CHECK(cudaDeviceGetAttribute(
                &num_SM, cudaDevAttrMultiProcessorCount, device_id));

        grids[device_id].first = max_threads_block;
        grids[device_id].second = max_threads_SM / max_threads_block * num_SM;
    }

    *block_size = grids[device_id].first;
    *grid_size = grids[device_id].second;
}

/*
    非启发式配置，线程块大小都选择256，线程块数为刚好能使整个gpu满载的配置，即每个sm可驻留的最
    大线程块数(由最大数量计算)乘上sm个数，对于目前的GPU，128也是一个不错的选择，但不能选择64，
    否则一个sm上需要驻留的线程块数可能超出限制
*/
inline void LaunchConfig(int *grid_size, int *block_size) {
    static int grids[MAX_DEVICES] = {0};       // 保存某个设备的线程块数设置

    *block_size = BLOCK_SIZE;

    int device_id = 0;
    cudaGetDevice(&device_id);
    if (device_id >= MAX_DEVICES)
        throw std::runtime_error{"设备数量过大，请调整最大设备数量"};

    if (grids[device_id] == 0) {
        int max_threads_SM = 0;
        int num_SM = 0;

        CHECK(cudaDeviceGetAttribute(
                &max_threads_SM, cudaDevAttrMaxThreadsPerMultiProcessor, device_id));
        CHECK(cudaDeviceGetAttribute(
                &num_SM, cudaDevAttrMultiProcessorCount, device_id));

        grids[device_id] = max_threads_SM / (*block_size) * num_SM;
    }

    *grid_size = grids[device_id];
}

/*
    执行kernel函数
    @param kernel: 核函数
    @param shared_mem: 一个线程块中需要分配的共享内存数量，字节为单位
    @param stream: cuda流
    @...args: 传递给核函数的参数
*/
// 在函数中计算线程网格大小
template<typename KernelFunction, typename ...Args>
void LaunchKernel(KernelFunction kernel, size_t shared_memory,
                  cudaStream_t stream, Args ...args) {
    int grid_size = 0;
    int block_size = 0;
    LaunchConfig(&grid_size, &block_size);
    kernel<<<grid_size, block_size, shared_memory, stream>>>(args...);
}

// 在函数中计算线程网格大小，指定如何根据线程块大小选择共享内存数量和流
template<typename KernelFunction, typename UnaryFunction, typename ...Args>
void LaunchKernelVariableSMem(KernelFunction kernel,
                              UnaryFunction BlockSizeToSMem,
                              cudaStream_t stream,
                              Args ...args) {
    int grid_size = 0;
    int block_size = 0;
    LaunchConfig(&grid_size, &block_size);
    kernel<<<grid_size, block_size, BlockSizeToSMem(block_size), stream>>>(args...);
}

// 需要指定线程网格大小的LaunchKernel
template<typename KernelFunction, typename ...Args>
void LaunchKernelWithGrid(KernelFunction kernel, int grid_size, int block_size,
                          size_t shared_memory, cudaStream_t stream, Args ...args) {
    kernel<<<grid_size, block_size, shared_memory, stream>>>(args...);
}

// 返回线程网格大小使得设备占用率最高，共享内存数量是线程块大小的一维函数
template<typename UnaryFunction, typename KernelFunction>
inline cudaError_t OccupancyMaxPotentialBlockSizeVariableSMem(
        int *min_grid_size,
        int *block_size,
        KernelFunction func,
        UnaryFunction BlockSizeToSMem,
        int block_size_limit = 0) {
    // Check user input
    if (!min_grid_size || !block_size || !func) {
        return cudaErrorInvalidValue;
    }

    cudaError_t status;

    // Device and function properties
    int device;
    struct cudaFuncAttributes attr;

    // Limits
    int max_threads_SM;
    int warp_size;
    int max_threads_block;
    int num_SM;
    int func_max_threads_per_block;
    int occupancy_limit;
    int granularity;

    // Recorded maximum
    int max_block_size = 0;
    int num_blocks = 0;
    int max_occupancy = 0;

    // Temporary
    int block_size_try_aligned;
    int block_size_try;
    int block_size_limit_aligned;
    int occupancy_in_blocks;
    int occupancy_in_threads;
    size_t dynamic_smem_size;

    // Obtain device and function properties
    status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaDeviceGetAttribute(
            &max_threads_SM,
            cudaDevAttrMaxThreadsPerMultiProcessor,
            device);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaDeviceGetAttribute(
            &warp_size,
            cudaDevAttrWarpSize,
            device);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaDeviceGetAttribute(
            &max_threads_block,
            cudaDevAttrMaxThreadsPerBlock,
            device);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaDeviceGetAttribute(
            &num_SM,
            cudaDevAttrMultiProcessorCount,
            device);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaFuncGetAttributes(&attr, func);
    if (status != cudaSuccess) {
        return status;
    }

    func_max_threads_per_block = attr.maxThreadsPerBlock;

    // Try each block size, and pick the block size with maximum occupancy
    occupancy_limit = max_threads_SM;
    granularity = warp_size;

    if (block_size_limit == 0) {
        block_size_limit = max_threads_block;
    }

    if (max_threads_block < block_size_limit) {
        block_size_limit = max_threads_block;
    }

    if (func_max_threads_per_block < block_size_limit) {
        block_size_limit = func_max_threads_per_block;
    }

    block_size_limit_aligned = ((block_size_limit + (granularity - 1)) / granularity) * granularity;

    for (block_size_try_aligned = block_size_limit_aligned;
         block_size_try_aligned > 0; block_size_try_aligned -= granularity) {
        // This is needed for the first iteration, because
        // block_size_limit_aligned could be greater than block_size_limit

        if (block_size_limit < block_size_try_aligned) {
            block_size_try = block_size_limit;
        } else {
            block_size_try = block_size_try_aligned;
        }

        dynamic_smem_size = BlockSizeToSMem(block_size_try);

        status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &occupancy_in_blocks,
                func,
                block_size_try,
                dynamic_smem_size);

        if (status != cudaSuccess) {
            return status;
        }

        occupancy_in_threads = block_size_try * occupancy_in_blocks;

        if (occupancy_in_threads > max_occupancy) {
            max_block_size = block_size_try;
            num_blocks = occupancy_in_blocks;
            max_occupancy = occupancy_in_threads;
        }

        // Early out if we have reached the maximum
        //
        if (occupancy_limit == max_occupancy) {
            break;
        }
    }

    // Return best available
    *min_grid_size = num_blocks * num_SM;
    *block_size = max_block_size;

    return status;
}

// 返回线程网格大小使得设备占用率最高，共享内存数量是常数
template<typename KernelFunction>
inline cudaError_t OccupancyMaxPotentialBlockSize(
        int *min_grid_size,
        int *block_size,
        KernelFunction func,
        size_t dynamic_smem_size = 0,
        int block_size_limit = 0) {
    return OccupancyMaxPotentialBlockSizeVariableSMem(
            min_grid_size,
            block_size,
            func,
            [dynamic_smem_size](int block_size) { return dynamic_smem_size; },
            block_size_limit);
}

#endif // LAUNCH_H

