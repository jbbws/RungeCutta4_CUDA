
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>


cudaError_t RungeCutta4(const int size,float step, float a, float b, float *init, float *y1, float *y2, float *y3);

//Элементы константной памяти
__device__ __constant__ float h;
__device__ __constant__ float l;
__device__ __constant__ float r;
//Переменная используемая ядром
__device__ int arraySize = 0;
//вспомогтальная функция вычисления значения функции  
__device__ float Func(const int number,float x, float y1, float y2, float y3)
{
	switch (number)
	{
	case 0:
		return -(55 + y3)*y1 + 65 * y2;
		break;
	case 1:
		return 0.0785*(y1-y2);
		break;
	case 2:
		return 0.1*y1;
		break;
	default:
		break;
	}
}
//функция вывода в консоль
__host__ void showResult(const float  *y1res,const float *y2res,const float *y3res,int size)
{
	float x = 0.0;
	for (int i = 0; i < size; i++)
	{
		printf("Step: %.2f ",x);
		printf(" y1[%d]: %f", i, y1res[i]);
		printf(" y2[%d]: %f", i, y2res[i]);
		printf(" y3[%d]: %f\n", i, y3res[i]);
		x = x + 0.01;
	}
}
//Это пока что не работает
__host__ void mallocMemoryOnHost(float *y1, float *y2, float *y3,const int size)
{
	/*y1 = (float*)malloc(size * sizeof(float));
	y2 = (float*)malloc(size * sizeof(float));
	y3 = (float*)malloc(size * sizeof(float));*/
	y1 = new float[size];
	y2 = new float[size];
	y3 = new float[size];
}
__host__ void freeMemoryOnHost(float *y1, float *y2, float *y3)
{
	delete[]y1;
	delete[]y2;
	delete[]y3;
}
//Ядро
__global__ void core(float *y1, float *y2, float *y3)
{
	int id = threadIdx.x;
	float K0, K1, K2, K3;
	for (int i = 1, float x = l; i < arraySize; i++, x = x + h)
	{
		K0 = Func(id, x, y1[i - 1], y2[i - 1], y3[i - 1]);
		K1 = Func(id, x + h / 2, y1[i - 1] + h / 2 * K0, y2[i - 1] + h / 2 * K0, y3[i - 1] + h / 2 * K0);
		K2 = Func(id, x + h / 2, y1[i - 1] + h / 2 * K1, y2[i - 1] + h / 2 * K1, y3[i - 1] + h / 2 * K1);
		K3 = Func(id, x + h, y1[i - 1] + h * K2, y2[i - 1] + h * K2, y3[i - 1] + h * K2);
		__syncthreads();
		switch (id)
		{
		case 0:
			y1[i] = y1[i - 1] + h / 6 * (K0 + 2 * K1 + 2 * K2 + K3);
			break;
		case 1:
			y2[i] = y2[i - 1] + h / 6 * (K0 + 2 * K1 + 2 * K2 + K3);
			break;
		case 2:
			y3[i] = y3[i - 1] + h / 6 * (K0 + 2 * K1 + 2 * K2 + K3);
			break;
		default:
			break;
		}
		__syncthreads();
	}
}
int main()
{
	
	const int countEq = 3;
	const float step = 0.01;
	const float a = 0;
	const float b = 0.1;

	float *y1 = 0, *y2 = 0, *y3 = 0;
	float initValues[countEq] = { 1.0f,1.0f,0.0f };
	int arrSize = static_cast<int>((b - a) / step) + 1;

	//mallocMemoryOnHost(y1, y2, y3, arrSize);
	y1 = new float[arrSize];
	y2 = new float[arrSize];
	y3 = new float[arrSize];

	cudaError_t CudaStat = 	RungeCutta4(arrSize,step,a,b,initValues,y1,y2,y3);
	if (CudaStat != cudaSuccess)
	{
		fprintf(stderr, "RungeCutta4 failed!");
		return 1;
	}
	CudaStat = cudaDeviceReset();
	if (CudaStat != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	showResult(y1, y2, y3, arrSize);

	delete[]y1;
	delete[]y2;
	delete[]y3;
	//freeMemoryOnHost(y1, y2, y3);
	getchar();
	return 0;
}

cudaError_t	RungeCutta4(const int size, float step,float a, float b,float *init, float *y1, float *y2, float *y3)
{
	float *dev_y1 = 0;				//массив для 1 уравнения
	float *dev_y2 = 0;				//массив для 2 уравнения
	float *dev_y3 = 0;				//массив для 3 уравнения
	cudaError_t	status;
	//printf("Size is: %d", size);
	status = cudaSetDevice(0);	//выбираем устройство
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	status = cudaMalloc((void**)&dev_y1, size * sizeof(float));			//выделили память для 1-ого массива
	status = cudaMalloc((void**)&dev_y2, size * sizeof(float));			//выделили память для 2-ого массива
	status = cudaMalloc((void**)&dev_y3, size * sizeof(float));			//выделили память для 3-ого массива
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	status = cudaMemcpyToSymbol(h, &step, sizeof(float), 0, cudaMemcpyHostToDevice);//копируем в конст.память значение шага
	status = cudaMemcpyToSymbol(l, &a, sizeof(float), 0, cudaMemcpyHostToDevice);//копируем в конст.память значение левой границы
	status = cudaMemcpyToSymbol(r, &b, sizeof(float), 0, cudaMemcpyHostToDevice);//копируем в конст.память значение правой границы
	status = cudaMemcpyToSymbol(arraySize, &size, sizeof(int), 0, cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyToSymbol failed!");
		goto Error;
	}
	//Копируем в память начальные значения для задачи КОШИ
	status = cudaMemcpy(dev_y1, &init[0], sizeof(float), cudaMemcpyHostToDevice); 
	status = cudaMemcpy(dev_y2, &init[1], sizeof(float), cudaMemcpyHostToDevice);
	status = cudaMemcpy(dev_y3, &init[2], sizeof(float), cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy init values failed!");
		goto Error;
	}

	core<<<1, 3 >>> (dev_y1, dev_y2, dev_y3);
	status = cudaGetLastError();
	if (status != cudaSuccess) {
		fprintf(stderr, "core launch failed: %s\n", cudaGetErrorString(status));
		goto Error;
	}
	status = cudaDeviceSynchronize();
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", status);
		goto Error;
	}
	status = cudaMemcpy(y1, dev_y1, size * sizeof(float), cudaMemcpyDeviceToHost);
	status = cudaMemcpy(y2, dev_y2, size * sizeof(float), cudaMemcpyDeviceToHost);
	status = cudaMemcpy(y3, dev_y3, size * sizeof(float), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyDeviceToHost failed!");
		goto Error;
	}
Error:
	cudaFree(dev_y1);
	cudaFree(dev_y2);
	cudaFree(dev_y3);
	return status;
}
