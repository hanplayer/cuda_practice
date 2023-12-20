
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#if 0
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

#endif



#if 0

__global__ void vecAddKernel(int* c, int* a, int* b,int n) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int a[] = { 1,2,3,4,5,6,7,8,9,10,11,12 };
    int b[] = { 1,2,3,4,5,6,7,8,9,10,11,12 };
    int size = sizeof(a);
    int c[12] = { 0 };
    int* cuda_A;
    int* cuda_B;
    int* cuda_C;
 
    //printf("size :%d",num);
    cudaMalloc((void**)&cuda_A, size);
    cudaMemcpy(cuda_A,a, size,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&cuda_B,size);
    cudaMemcpy(cuda_B,b,size,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&cuda_C, size);
    
    vecAddKernel<<<4, 4>>>(cuda_C,cuda_A,cuda_B,12);

    cudaMemcpy(c,cuda_C,48, cudaMemcpyDeviceToHost);

    for (auto i = 0; i < 12;i++) {
        printf("index:%d value:%d\n",i,c[i]);
    }
    cudaFree(cuda_A);
    cudaFree(cuda_B);
    cudaFree(cuda_C);

    return 0;
}

#endif


#if 0
//简单方阵相乘  

//请注意是方阵  所以只传一个

__global__ void MatrixMulKernel(int* a,int* b,int *c,int width) {
    int col = blockIdx.x * gridDim.x + threadIdx.x;
    int row = blockIdx.y * gridDim.y + threadIdx.y;

    int count = 0;
    if ((col < width) && (row < width)) {
        for (int i = 0; i < width;i++) {
            //count += a[row][i]*b[i][col];
            count += a[row * width + i] * b[i * width + col];
        }
        c[row*width+col] = count;
    }

}

int main() {

    int a[] = { 1,2,3,4,
               5,6,7,8,
               9,10,11,12,
               13,14,15,16 };
    int b[] = { 1,2,3,4,
               5,6,7,8,
               9,10,11,12,
               13,14,15,16 };
    //int b[] = { 1,1,1,1,
    //       1,1,1,1,
    //       1,1,1,1,
    //       1,1,1,1 };

    int c[16] = {};

    int size = sizeof(a);
    int * cuda_A;
    int * cuda_B;
    int * cuda_C;
    cudaMalloc(&cuda_A, size);
    cudaMemcpy(cuda_A,a,size, cudaMemcpyHostToDevice);
    cudaMalloc(&cuda_B, size);
    cudaMemcpy(cuda_B,b,size, cudaMemcpyHostToDevice);
    cudaMalloc(&cuda_C, size);

    dim3 dimGrid(2,2,1);
    dim3 dimBlock(2,2,1);

    MatrixMulKernel << <dimGrid, dimBlock >> > (cuda_A, cuda_B, cuda_C, 4);

    cudaMemcpy(c, cuda_C,size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 16;i++) {
        printf("idx:%d value:%d\n",i,c[i]);
    }
    cudaFree(cuda_A);
    cudaFree(cuda_B);
    cudaFree(cuda_C);

    return 0;
}

#endif