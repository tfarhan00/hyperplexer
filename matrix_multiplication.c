#include <Python.h>
#include <stdlib.h>
#include <arm_neon.h>
#include <string.h>
#include <pthread.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define THREADS 4

typedef struct {
    int32_t *mat1;
    int32_t *mat2;
    int32_t *result;
    Py_ssize_t n;
    Py_ssize_t m;
    Py_ssize_t p;
    Py_ssize_t start;
    Py_ssize_t end;
} ThreadData;

// Function to convert Python list to C array
static int32_t* pylist_to_carray(PyObject* py_list, Py_ssize_t* rows, Py_ssize_t* cols) {
    *rows = PyList_Size(py_list);
    *cols = PyList_Size(PyList_GetItem(py_list, 0));

    int32_t* c_array = (int32_t*)aligned_alloc(32, (*rows) * (*cols) * sizeof(int32_t));
    for (Py_ssize_t i = 0; i < *rows; i++) {
        PyObject* row = PyList_GetItem(py_list, i);
        for (Py_ssize_t j = 0; j < *cols; j++) {
            c_array[i*(*cols) + j] = (int32_t)PyLong_AsLong(PyList_GetItem(row, j));
        }
    }

    return c_array;
}

// Function to convert C array to Python list
static PyObject* carray_to_pylist(int32_t* c_array, Py_ssize_t rows, Py_ssize_t cols) {
    PyObject* py_list = PyList_New(rows);
    for (Py_ssize_t i = 0; i < rows; i++) {
        PyObject* row = PyList_New(cols);
        for (Py_ssize_t j = 0; j < cols; j++) {
            PyList_SetItem(row, j, PyLong_FromLong(c_array[i*cols + j]));
        }
        PyList_SetItem(py_list, i, row);
    }

    return py_list;
}

// Optimized matrix multiplication function with NEON intrinsics
static void* matrix_multiply_neon(void* arg) {
    ThreadData *data = (ThreadData*)arg;
    int32_t *mat1 = data->mat1;
    int32_t *mat2 = data->mat2;
    int32_t *result = data->result;
    Py_ssize_t n = data->n;
    Py_ssize_t m = data->m;
    Py_ssize_t p = data->p;
    Py_ssize_t start = data->start;
    Py_ssize_t end = data->end;
    
    const int BLOCK_SIZE = 64;
    
    for (Py_ssize_t ii = start; ii < end; ii += BLOCK_SIZE) {
        for (Py_ssize_t jj = 0; jj < p; jj += BLOCK_SIZE) {
            for (Py_ssize_t kk = 0; kk < m; kk += BLOCK_SIZE) {
                Py_ssize_t max_i = MIN(ii + BLOCK_SIZE, end);
                Py_ssize_t max_j = MIN(jj + BLOCK_SIZE, p);
                Py_ssize_t max_k = MIN(kk + BLOCK_SIZE, m);

                for (Py_ssize_t i = ii; i < max_i; i++) {
                    for (Py_ssize_t j = jj; j < max_j; j++) {
                        int32x4_t sum_vec = vdupq_n_s32(0);
                        for (Py_ssize_t k = kk; k < max_k; k += 4) {
                            int32x4_t mat1_vec = vld1q_s32(&mat1[i*m + k]);
                            int32x4_t mat2_vec = vld1q_s32(&mat2[k*p + j]);
                            sum_vec = vmlaq_s32(sum_vec, mat1_vec, mat2_vec);
                        }
                        int32x2_t sum_vec2 = vadd_s32(vget_low_s32(sum_vec), vget_high_s32(sum_vec));
                        result[i*p + j] += vget_lane_s32(vpadd_s32(sum_vec2, sum_vec2), 0);
                    }
                }
            }
        }
    }
    return NULL;
}

// Function to perform matrix multiplication
static PyObject* matrix_multiply(PyObject* self, PyObject* args) {
    PyObject *py_mat1, *py_mat2;
    if (!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &py_mat1, &PyList_Type, &py_mat2)) {
        return NULL;
    }

    Py_ssize_t n, m, p;
    int32_t* mat1 = pylist_to_carray(py_mat1, &n, &m);
    int32_t* mat2 = pylist_to_carray(py_mat2, &m, &p);

    int32_t* result = (int32_t*)aligned_alloc(32, n * p * sizeof(int32_t));
    memset(result, 0, n * p * sizeof(int32_t));

    pthread_t threads[THREADS];
    ThreadData thread_data[THREADS];
    Py_ssize_t chunk_size = n / THREADS;

    for (int i = 0; i < THREADS; i++) {
        thread_data[i].mat1 = mat1;
        thread_data[i].mat2 = mat2;
        thread_data[i].result = result;
        thread_data[i].n = n;
        thread_data[i].m = m;
        thread_data[i].p = p;
        thread_data[i].start = i * chunk_size;
        thread_data[i].end = (i == THREADS - 1) ? n : (i + 1) * chunk_size;
        pthread_create(&threads[i], NULL, matrix_multiply_neon, &thread_data[i]);
    }

    for (int i = 0; i < THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    PyObject* py_result = carray_to_pylist(result, n, p);

    free(mat1);
    free(mat2);
    free(result);

    return py_result;
}

// Method definition object for this extension, defining the function(s)
static PyMethodDef MatrixMethods[] = {
    {"matrix_multiply", (PyCFunction)matrix_multiply, METH_VARARGS,
     "matrix_multiply(mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]\n"
     "Multiply two matrices.\n\n"
     "Parameters:\n"
     "  mat1 (List[List[int]]): The first matrix.\n"
     "  mat2 (List[List[int]]): The second matrix.\n\n"
     "Returns:\n"
     "  List[List[int]]: The result of the matrix multiplication."},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef matrixmodule = {
    PyModuleDef_HEAD_INIT,
    "hyperplexer",
    "A module for performing matrix multiplication.",
    -1,
    MatrixMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_matrix_module(void) {
    return PyModule_Create(&matrixmodule);
}
