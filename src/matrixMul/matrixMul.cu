/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication as described in Chapter 3
 * of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * CUBLAS provides high-performance matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Superconducting (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11. 
 *
 */

// Utilities and system includes
#include <cublas_v2.h>
#include <shrUtils.h>
#include <shrQATest.h>
#include "cutil_inline.h"
#include "matrixMul.h"

// includes, kernels
#include "matrixMul_kernel.cu"

static char *sSDKsample = "matrixMul";

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);
void randomInit(float*, int);
void printDiff(float*, float*, int, int, int, float);

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

void inline checkError(cublasStatus_t status, const char* msg)
{
    if(status != CUBLAS_STATUS_SUCCESS){
        printf(msg);
        exit(-1);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    shrQAStart(argc, argv);
	printf("[ %s ]\n", sSDKsample);

    //shrSetLogFileName ("matrixMul.txt");
    shrLog("%s Starting (CUDA and CUBLAS tests)...\n\n", argv[0]);

    runTest(argc, argv);
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
        cutilSafeCall( cudaSetDevice(cutGetMaxGflopsDeviceId()) );
    }

    int devID;
    cudaDeviceProp props;

    // get number of SMs on this GPU
    cutilSafeCall(cudaGetDevice(&devID));
    cutilSafeCall(cudaGetDeviceProperties(&props, devID));

    // use a larger block size for Fermi and above
    int block_size = (props.major < 2) ? 16 : 32;

    printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);

	// set seed for rand()
    srand(2006);

    // Optional Command-line multiplier for matrix sizes
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
    int iSizeMultiple = 5;
    shrGetCmdLineArgumenti(argc, (const char**)argv, "sizemult", &iSizeMultiple); 
    iSizeMultiple = CLAMP(iSizeMultiple, 1, 10);

    bool useCublasOnly = false;
    if(shrCheckCmdLineFlag(argc, (const char**)argv, "cublas"))
        useCublasOnly = true;

	// For GPUs with fewer # of SM's, we limit the maximum size of the matrix
	if (props.multiProcessorCount <= 4) {
		uiWA = 2 * block_size * iSizeMultiple;
		uiHA = 4 * block_size * iSizeMultiple;
		uiWB = 2 * block_size * iSizeMultiple;
		uiHB = 4 * block_size * iSizeMultiple;
		uiWC = 2 * block_size * iSizeMultiple;
		uiHC = 4 * block_size * iSizeMultiple;
	} else {
		uiWA = WA * iSizeMultiple;
		uiHA = HA * iSizeMultiple;
		uiWB = WB * iSizeMultiple;
		uiHB = HB * iSizeMultiple;
		uiWC = WC * iSizeMultiple;
		uiHC = HC * iSizeMultiple;
	}
    shrLog("\nUsing Matrix Sizes: A(%u x %u), B(%u x %u), C(%u x %u)\n\n", 
            uiWA, uiHA, uiWB, uiHB, uiWC, uiHC);

    // allocate host memory for matrices A and B
    unsigned int size_A = uiWA * uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*)malloc(mem_size_A);
    unsigned int size_B = uiWB * uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*)malloc(mem_size_B);

    // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);
    
    // allocate device memory
    float* d_A, *d_B, *d_C;
    unsigned int size_C = uiWC * uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;

    // allocate host memory for the result
    float* h_C      = (float*) malloc(mem_size_C);
	float* h_CUBLAS = (float*) malloc(mem_size_C);

    cutilSafeCall(cudaMalloc((void**) &d_A, mem_size_A));
    cutilSafeCall(cudaMalloc((void**) &d_B, mem_size_B));

    // copy host memory to device
    cutilSafeCall(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice) );
    cutilSafeCall(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice) );
    
    cutilSafeCall(cudaMalloc((void**) &d_C, mem_size_C));
   
    // setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(uiWC / threads.x, uiHC / threads.y);

    // kernel warmup
    if(useCublasOnly) {
	} else {
    }
    
    // create and start timer
    shrLog("Runing Kernels...\n\n");
    unsigned int timer_cublas    = 0;
    unsigned int timer_matrixMul = 0;

    // execute the kernel
    int nIter = 30;

	// CUBLAS version 2.0
	{
        cublasHandle_t handle;
        checkError(cublasCreate(&handle), "cublasCreate() error!\n");
        const float alpha = 1.0f;
        const float beta = 0.0f;
        //Perform warmup operation with cublas
        cublasStatus_t ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, uiWB, uiHA, uiWA, &alpha, d_B, uiWB, d_A, uiWA, &beta, d_C, uiWA);
        checkError(ret, "cublas Sgemm returned an error!\n");

		// Start Timing
		cutilCheckError(cutCreateTimer(&timer_cublas));
		cutilCheckError(cutStartTimer(timer_cublas));
        for (int j = 0; j < nIter; j++) {
            //note cublas is column primary!
            //need to transpose the order
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, uiWB, uiHA, uiWA, &alpha, d_B, uiWB, d_A, uiWA, &beta, d_C, uiWA);
		}
		// check if kernel execution generated and error
		cutilCheckMsg("CUBLAS Kernel execution failed");
		cutilDeviceSynchronize();
		// stop and destroy timer
		cutilCheckError(cutStopTimer(timer_cublas));

		double dSeconds = cutGetTimerValue(timer_cublas)/((double)nIter * 1000.0);
		double dNumOps = 2.0 * (double)uiWA * (double)uiHA * (double)uiWB;
		double gflops = 1.0e-9 * dNumOps/dSeconds;

		//Log througput, etc
		shrLogEx(LOGBOTH | MASTER, 0, "> CUBLAS         Throughput = %.4f GFlop/s, Time = %.5f s, Size = %.0f Ops\n\n", 
				gflops, dSeconds, dNumOps);

		cutilCheckError(cutDeleteTimer(timer_cublas));

		// copy result from device to host
		cutilSafeCall(cudaMemcpy(h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost) );

        checkError(cublasDestroy(handle), "cublasDestroy() error!\n");
	}

	// For the case where "-cublas" is not specified, we will run the matrixMul kernel
	if (!useCublasOnly) 
	{
        //Performs warmup operation using matrixMul CUDA kernel
		if (block_size == 16) {
            matrixMul<16><<< grid, threads >>>(d_C, d_A, d_B, uiWA, uiWB);
        } else {
            matrixMul<32><<< grid, threads >>>(d_C, d_A, d_B, uiWA, uiWB);
        }
        cutilDeviceSynchronize();

		// Start Timing	
		cutilCheckError(cutCreateTimer(&timer_matrixMul));
		cutilCheckError(cutStartTimer(timer_matrixMul));
		for (int j = 0; j < nIter; j++) {
			if (block_size == 16) {
				matrixMul<16><<< grid, threads >>>(d_C, d_A, d_B, uiWA, uiWB);
			} else {
				matrixMul<32><<< grid, threads >>>(d_C, d_A, d_B, uiWA, uiWB);
			}
		}
		// check if kernel execution generated and error
		cutilCheckMsg("CUDA matrixMul Kernel execution failed");

        cutilDeviceSynchronize();
		// stop and destroy timer
		cutilCheckError(cutStopTimer(timer_matrixMul));

		double dSeconds = cutGetTimerValue(timer_matrixMul)/((double)nIter * 1000.0);
		double dNumOps = 2.0 * (double)uiWA * (double)uiHA * (double)uiWB;
		double gflops = 1.0e-9 * dNumOps/dSeconds;

		//Log througput, etc
		shrLogEx(LOGBOTH | MASTER, 0, "> CUDA matrixMul Throughput = %.4f GFlop/s, Time = %.5f s, Size = %.0f Ops, ", 
				gflops, dSeconds, dNumOps);
		shrLogEx(LOGBOTH | MASTER, 0, "NumDevsUsed = %d, Workgroup = %u\n", 1, threads.x * threads.y);

		cutilCheckError(cutDeleteTimer(timer_matrixMul));

		// copy result from device to host
		cutilSafeCall(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost) );
	}

    // compute reference solution
    shrLog("\nComparing GPU results with Host computation...\n\n");    
    float* reference = (float*)malloc(mem_size_C);
    computeGold(reference, h_A, h_B, uiHA, uiWA, uiWB);

    // check result (CUBLAS)
	printf("Comparing CUBLAS & Host results\n");
    shrBOOL resCUBLAS = shrCompareL2fe(reference, h_CUBLAS, size_C, 1.0e-6f);
    if (resCUBLAS != shrTRUE) 
    {
        printDiff(reference, h_CUBLAS, uiWC, uiHC, 100, 1.0e-5f);
    }
    shrLog("CUBLAS compares %s\n\n", (shrTRUE == resCUBLAS) ? "OK" : "FAIL");

    // check result (matrixMul)
	printf("Comparing CUDA matrixMul & Host results\n");
    shrBOOL resCUDA = shrCompareL2fe(reference, h_C, size_C, 1.0e-6f);
    if (resCUDA != shrTRUE) 
    {
        printDiff(reference, h_C, uiWC, uiHC, 100, 1.0e-5f);
    }
    shrLog("CUDA matrixMul compares %s\n\n", (shrTRUE == resCUDA) ? "OK" : "FAIL");

    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(reference);
    cutilSafeCall(cudaFree(d_A));
    cutilSafeCall(cudaFree(d_B));
    cutilSafeCall(cudaFree(d_C));

    cutilDeviceReset();
    shrQAFinishExit(argc, (const char **)argv, (resCUDA == shrTRUE && resCUBLAS == shrTRUE) ? QA_PASSED : QA_FAILED);
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    shrLog("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;
    for (j = 0; j < height; j++) 
    {
        if (error_count < iListLength)
        {
            shrLog("\n  Row %d:\n", j);
        }
        for (i = 0; i < width; i++) 
        {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);
            if (fDiff > fListTol) 
            {                
                if (error_count < iListLength)
                {
                    shrLog("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }
                error_count++;
            }
        }
    }
    shrLog(" \n  Total Errors = %d\n\n", error_count);
}
