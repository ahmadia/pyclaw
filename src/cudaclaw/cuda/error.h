#define CHKERR() if (cudaPeekAtLastError()) { printf("CUDA error in %s at %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaPeekAtLastError())); return cudaPeekAtLastError(); }

#define CHKERRQ(n) if (n) { printf("CUDA error in %s at %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(n)); return n; }

#define CHKERR_ABORT() if (cudaPeekAtLastError()) { printf("CUDA error in %s at %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaPeekAtLastError())); }

#define CHKERRQ_ABORT(n) if (n) { printf("CUDA error in %s at %d: %s\nIgnoring!\n", __FILE__, __LINE__, cudaGetErrorString(n)); }
