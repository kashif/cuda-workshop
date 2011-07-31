// Utilities and system includes
#include <shrUtils.h>
#include <cutil_inline.h>

// includes, kernels
#include "quadFloat_kernel.cu"

unsigned int timer = 0;

static char *sSDKsample = "quadFloat";

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    printf("[ %s ]\n", sSDKsample);

    shrSetLogFileName ("quadFloat.txt");
    shrLog("%s Starting...\n\n", argv[0]);

    runTest(argc, argv);

    shrEXIT(argc, (const char**)argv);
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char** argv)
{
    if(shrCheckCmdLineFlag(argc, (const char**)argv, "device"))
    {
        cutilDeviceInit(argc, argv);
    }
    else
    {
        cudaSetDevice(cutGetMaxGflopsDeviceId());
    }

    int devID;
    cudaDeviceProp props;

    // get number of SMs on this GPU
    cutilSafeCall(cudaGetDevice(&devID));
    cutilSafeCall(cudaGetDeviceProperties(&props, devID));

    printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);
    
    // setup execution parameters
    dim3 threads(256, 256);
    dim3 grid(32 / threads.x, 32 / threads.y);
    
    // allocate device memory
    uint* d_A;
    cutilSafeCall(cudaMalloc((void**) &d_A, 256 * 256 * sizeof(uint)));
    
    cutilCheckError( cutCreateTimer( &timer));
    
    cutStartTimer(timer);
    
    qfComputeMandelbrot<<< grid, threads >>>(d_A, 256, 256, 1, 1, 256, 256, make_float2(1.0,1.0), make_float2(1.0,1.0), make_float2(0.5,0.5), make_float2(0.5, 0.5), 1000);
    
    cudaThreadSynchronize();
    cutStopTimer(timer);

    double dAvgTime = cutGetTimerValue(timer)/1000.0;
    
    printf("Time for quadFloat = %.5f\n", dAvgTime);
    
    // allocate host memory for the result
    uint* h_A = (uint*) malloc(256 * 256 * sizeof(uint));
    
    // copy result from device to host
    cutilSafeCall(cudaMemcpy(h_A, d_A, 256 * 256 * sizeof(uint),
                              cudaMemcpyDeviceToHost) );
                              
    printf("%u", h_A[0]);
                              
    cutilCheckError( cutDeleteTimer( timer));
    cutilSafeCall(cudaFree(d_A)); 
}
