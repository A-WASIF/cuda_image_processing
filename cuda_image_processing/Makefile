NVCC = nvcc
CCBIN = g++-11

all:
	$(NVCC) -ccbin $(CCBIN) src/main.cu src/gpu_kernels.cu -o main `pkg-config --cflags --libs opencv4`

clean:
	rm -f main