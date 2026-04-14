
#include <cuda_runtime.h>
#include <math.h>

__global__ void rgbToGrayscale(unsigned char* input, unsigned char* output, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx < total) {
        int i = idx * 3;
        output[idx] = 0.299f*input[i] + 0.587f*input[i+1] + 0.114f*input[i+2];
    }
}

__global__ void sobel(unsigned char* input, unsigned char* output, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x>0 && y>0 && x<w-1 && y<h-1) {
        int gx = -input[(y-1)*w+(x-1)] + input[(y-1)*w+(x+1)]
               -2*input[y*w+(x-1)]     + 2*input[y*w+(x+1)]
               -input[(y+1)*w+(x-1)] + input[(y+1)*w+(x+1)];

        int gy = -input[(y-1)*w+(x-1)] -2*input[(y-1)*w+x] -input[(y-1)*w+(x+1)]
               + input[(y+1)*w+(x-1)] +2*input[(y+1)*w+x] +input[(y+1)*w+(x+1)];

        int mag = sqrtf(gx*gx + gy*gy);
        output[y*w+x] = mag > 255 ? 255 : mag;
    }
}
