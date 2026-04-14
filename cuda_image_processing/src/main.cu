#include <iostream>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace std;
namespace fs = std::filesystem;

__global__ void rgbToGrayscale(unsigned char*, unsigned char*, int, int);
__global__ void sobel(unsigned char*, unsigned char*, int, int);

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: ./main input_dir output_dir\n";
        return 1;
    }

    string inputDir = argv[1];
    string outputDir = argv[2];

    int processed = 0;

    for (const auto &entry : fs::directory_iterator(inputDir)) {
        string path = entry.path().string();

        cv::Mat img = cv::imread(path);
        if (img.empty()) continue;

        int width = img.cols;
        int height = img.rows;

        cv::Mat gray(height, width, CV_8UC1);
        cv::Mat edge(height, width, CV_8UC1);

        unsigned char *d_input, *d_gray, *d_edge;

        cudaMalloc(&d_input, width * height * 3);
        cudaMalloc(&d_gray, width * height);
        cudaMalloc(&d_edge, width * height);

        cudaMemcpy(d_input, img.data, width * height * 3, cudaMemcpyHostToDevice);

        int block = 256;
        int grid = (width * height + block - 1) / block;

        rgbToGrayscale<<<grid, block>>>(d_input, d_gray, width, height);

        dim3 b2(16,16);
        dim3 g2((width+15)/16, (height+15)/16);

        sobel<<<g2, b2>>>(d_gray, d_edge, width, height);

        cudaMemcpy(edge.data, d_edge, width * height, cudaMemcpyDeviceToHost);

        string outPath = outputDir + "/out_" + to_string(processed) + ".png";
        cv::imwrite(outPath, edge);

        cudaFree(d_input);
        cudaFree(d_gray);
        cudaFree(d_edge);

        processed++;
    }

    cout << "Processed " << processed << " images using GPU\n";
    return 0;
}