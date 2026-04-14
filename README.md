# CUDA Image Processing Pipeline

## 📌 Project Overview

This project implements a GPU-accelerated image processing pipeline using CUDA. The program processes a large batch of images and applies multiple transformations in parallel on the GPU.

The pipeline includes:

* Grayscale conversion (custom CUDA kernel)
* Sobel edge detection (custom CUDA kernel)

The goal is to demonstrate how GPU parallelism significantly speeds up pixel-wise image operations compared to CPU execution.

---

## ⚙️ Technologies Used

* CUDA (GPU programming)
* OpenCV (image loading and saving)
* C++ (host-side logic)

---

## 📂 Dataset

The input dataset consists of **multiple TIFF images** stored in:

```
data/input/
```

The program processes all images in this directory.

---

## 🚀 How to Build

```bash
make
```

---

## ▶️ How to Run

```bash
./run.sh
```

---

## 📊 Output

* Processed images are saved in:

```
data/output/
```

* Execution logs are saved in:

```
logs/execution_log.txt
```

---

## 🧠 Implementation Details

### 1. Grayscale Conversion (CUDA Kernel)

Each thread processes one pixel:

* Converts RGB → grayscale using weighted sum

### 2. Sobel Edge Detection (CUDA Kernel)

* Applies Sobel operator to detect edges
* Computes gradient magnitude using X and Y filters

---

## ⚡ GPU Acceleration

* Each pixel is processed in parallel using CUDA threads
* Significant speedup achieved for large image batches
* Demonstrates data-parallel workload optimization

---

## 📈 Results

* Successfully processed **multiple images in a single execution**
* GPU handled pixel computations in parallel
* Output images clearly show detected edges

---

## 🧩 Challenges Faced

* CUDA compiler compatibility issues (GCC vs Clang)
* Handling image formats (TIFF support via OpenCV)
* Managing GPU memory (allocation and transfer)

---

## ✅ Conclusion

This project demonstrates how CUDA can be used to efficiently process large-scale image datasets by leveraging GPU parallelism.

---
