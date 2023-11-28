#include "prisma/core/tensor.h"

static prsm_tensor_t *prsm_tensor_mul_vec_by_mat(prsm_tensor_t *const out, const prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs);
static prsm_tensor_t *prsm_tensor_mul_mat_by_vec(prsm_tensor_t *const out, const prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs);
static prsm_tensor_t *prsm_tensor_mul_mat_by_mat(prsm_tensor_t *const out, const prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs);

/* 
    Tensor creation/destruction
*/

prsm_tensor_t *prsm_tensor_create(struct VitaBaseAllocatorType *const alloctr, const size_t ndim, ...) {
    // check for invalid input
    VT_DEBUG_ASSERT(ndim > 0, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // allocate for shape
    size_t *shape = (alloctr == NULL)
        ? VT_CALLOC(ndim * sizeof(size_t))
        : VT_ALLOCATOR_ALLOC(alloctr, ndim * sizeof(size_t));

    // find shape
    size_t total_size = 1;
    va_list args; va_start(args, ndim);
    VT_FOREACH(i, 0, ndim) {
        shape[i] = va_arg(args, size_t);
        total_size *= shape[i];
    }
    va_end(args);

    // allocate for data
    prsm_float *data = (alloctr == NULL)
        ? VT_CALLOC(total_size * sizeof(prsm_float))
        : VT_ALLOCATOR_ALLOC(alloctr, total_size * sizeof(prsm_float));

    // allocate for tensor
    prsm_tensor_t *t = (alloctr == NULL)
        ? VT_CALLOC(sizeof(prsm_tensor_t))
        : VT_ALLOCATOR_ALLOC(alloctr, sizeof(prsm_tensor_t));
    
    // create tensor
    *t = (prsm_tensor_t) {
        .is_view = false,
        .ndim = ndim,
        .shape = shape,
        .data = data,
        .alloctr = alloctr
    };

    return t;
}

prsm_tensor_t *prsm_tensor_create_shape(struct VitaBaseAllocatorType *const alloctr, const size_t ndim, const size_t shape[]) {
    // check for invalid input
    VT_DEBUG_ASSERT(ndim > 0, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // allocate for shape
    size_t *shape_ = (alloctr == NULL)
        ? VT_CALLOC(ndim * sizeof(size_t))
        : VT_ALLOCATOR_ALLOC(alloctr, ndim * sizeof(size_t));
    vt_memcopy(shape_, shape, ndim * sizeof(size_t));
    
    // find total size
    size_t total_size = 1;
    VT_FOREACH(i, 0, ndim) {
        total_size *= shape[i];
    }

    // allocate for data
    prsm_float *data = (alloctr == NULL)
        ? VT_CALLOC(total_size * sizeof(prsm_float))
        : VT_ALLOCATOR_ALLOC(alloctr, total_size * sizeof(prsm_float));

    // allocate for tensor
    prsm_tensor_t *t = (alloctr == NULL)
        ? VT_CALLOC(sizeof(prsm_tensor_t))
        : VT_ALLOCATOR_ALLOC(alloctr, sizeof(prsm_tensor_t));
    
    // create tensor
    *t = (prsm_tensor_t) {
        .is_view = false,
        .ndim = ndim,
        .shape = shape_,
        .data = data,
        .alloctr = alloctr
    };

    return t;
}

prsm_tensor_t *prsm_tensor_create_vec(struct VitaBaseAllocatorType *const alloctr, const size_t len) {
    // check for invalid input
    VT_DEBUG_ASSERT(len > 0, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    return prsm_tensor_create(alloctr, 1, len);
}

prsm_tensor_t *prsm_tensor_create_mat(struct VitaBaseAllocatorType *const alloctr, const size_t rows, const size_t cols) {
    // check for invalid input
    VT_DEBUG_ASSERT(rows > 0, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(cols > 0, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    return prsm_tensor_create(alloctr, 2, rows, cols);
}

void prsm_tensor_destroy(prsm_tensor_t *t) {
    // check for invalid input
    VT_DEBUG_ASSERT(t != NULL, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // if tensor is view, skip
    if (t->is_view) {
        return;
    }

    // free shape, data, tensor
    (t->alloctr) ? VT_ALLOCATOR_FREE(t->alloctr, t->shape) : VT_FREE(t->shape);
    (t->alloctr) ? VT_ALLOCATOR_FREE(t->alloctr, t->data) : VT_FREE(t->data);
    (t->alloctr) ? VT_ALLOCATOR_FREE(t->alloctr, t) : VT_FREE(t);
}

/* 
    Tensor properties
*/

bool prsm_tensor_is_null(const prsm_tensor_t *const t) {
    return (t == NULL || t->shape == NULL || t->data == NULL);
}

size_t prsm_tensor_dim(const prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    return t->ndim;
}

const size_t *prsm_tensor_shape(const prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    return t->shape;
}

prsm_float *prsm_tensor_data(const prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    return t->data;
}

size_t prsm_tensor_size(const prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // calculate size from shape
    size_t size = 1;
    VT_FOREACH(i, 0, t->ndim) {
        size *= t->shape[i];
    }

    return size;
}

/* 
    Tensor data structure operations
*/

void prsm_tensor_resize(prsm_tensor_t *const t, const size_t ndim, ...) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(ndim > 0, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(!t->is_view, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_IS_VIEW));

    // calculate old tensor size
    const size_t total_size_old = prsm_tensor_size(t);

    // reallocate shape
    if (t->ndim > ndim) {
        t->shape = (t->alloctr == NULL)
            ? VT_REALLOC(t->shape, ndim * sizeof(size_t))
            : VT_ALLOCATOR_REALLOC(t->alloctr, t->shape, ndim * sizeof(size_t));
    }
    t->ndim = ndim;

    // find new shape size
    size_t total_size = 1;
    va_list args; va_start(args, ndim);
    VT_FOREACH(i, 0, ndim) {
        t->shape[i] = va_arg(args, size_t);
        total_size *= t->shape[i];
    }
    va_end(args);

    // reallocate data
    if (total_size > total_size_old) {
        t->data = (t->alloctr == NULL)
            ? VT_CALLOC(total_size * sizeof(prsm_float))
            : VT_ALLOCATOR_ALLOC(t->alloctr, total_size * sizeof(prsm_float));
    }
}

void prsm_tensor_resize_shape(prsm_tensor_t *const t, const size_t ndim, const size_t shape[])  {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(ndim > 0, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(!t->is_view, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_IS_VIEW));

    // calculate old tensor size
    const size_t total_size_old = prsm_tensor_size(t);

    // reallocate shape
    if (t->ndim > ndim) {
        t->shape = (t->alloctr == NULL)
            ? VT_REALLOC(t->shape, ndim * sizeof(size_t))
            : VT_ALLOCATOR_REALLOC(t->alloctr, t->shape, ndim * sizeof(size_t));
    }
    t->ndim = ndim;

    // copy shape
    vt_memcopy(t->shape, shape, ndim * sizeof(size_t));

    // find new shape size
    size_t total_size = 1;
    VT_FOREACH(i, 0, ndim) {
        total_size *= t->shape[i];
    }

    // reallocate data
    if (total_size > total_size_old) {
        t->data = (t->alloctr == NULL)
            ? VT_CALLOC(total_size * sizeof(prsm_float))
            : VT_ALLOCATOR_ALLOC(t->alloctr, total_size * sizeof(prsm_float));
    }
}

prsm_tensor_t *prsm_tensor_dup(const prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // duplicate tensor
    prsm_tensor_t *tdup = prsm_tensor_create_shape(t->alloctr, t->ndim, t->shape);
    vt_memcopy(tdup->data, t->data, prsm_tensor_size(tdup) * sizeof(prsm_float));

    return tdup;
}

void prsm_tensor_dup_into(prsm_tensor_t *const out, const prsm_tensor_t *const in) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(out), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(in), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(prsm_tensor_match_shape(out, in), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_SHAPES));
    
    // copy data
    vt_memcopy(out->data, in->data, prsm_tensor_size(out) * sizeof(prsm_float));
}

void prsm_tensor_transpose(prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // transpose
    if (t->ndim == 1) {
        return;
    } else if (t->ndim == 2) {
        // transpose the matrix (from RosettaCode, the C version, uses permutations)
        // RosettaCode: http://www.rosettacode.org/wiki/Matrix_transposition#C
        // Wiki article on permutations: https://en.wikipedia.org/wiki/In-place_matrix_transposition
        const size_t r = t->shape[0];
        const size_t c = t->shape[1];
        VT_FOREACH(start, 0, r*c) {
            size_t next = start;
            size_t i = 0;

            do {
                i++;
                next = (next % r) * c + next / r;
            } while (next > start);

            if(next < start || i == 1) {
                continue;
            }

            const size_t tmp = t->data[next = start];
            do {
                i = (next % r) * c + next / r;
                t->data[next] = (i == start) ? tmp : t->data[i];
                next = i;
            } while (next > start);
        }

        // update tensor matrix shape
        t->shape[0] = c;
        t->shape[1] = r;
    } else {
        VT_UNIMPLEMENTED("Unsupported for now!");
    }
}

/* 
    Tensor data operations
*/

bool prsm_tensor_match_shape(const prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(lhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(rhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    return (lhs->ndim == rhs->ndim) && vt_memcmp(lhs->shape, rhs->shape, lhs->ndim * sizeof(size_t));
}

bool prsm_tensor_match_shape_ex(const prsm_tensor_t *const t, const size_t ndim, const size_t shape[]) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(shape != NULL, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    return (t->ndim == ndim) && vt_memcmp(t->shape, shape, t->ndim * sizeof(size_t));
}

bool prsm_tensor_equals(const prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(lhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(rhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    if (lhs->ndim != rhs->ndim) return false;
    if (prsm_tensor_size(lhs) != prsm_tensor_size(rhs)) return false;

    // check values
    VT_FOREACH(i, 0, prsm_tensor_size(lhs)) {
        if (lhs->data[i] != rhs->data[i]) return false;
    }

    return true;
}

void prsm_tensor_assign(prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(lhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(rhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(prsm_tensor_match_shape(lhs, rhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_SHAPES));

    // copy data
    vt_memcopy(lhs->data, rhs->data, prsm_tensor_size(lhs) * sizeof(prsm_float));
}

void prsm_tensor_assign_array(prsm_tensor_t *t, const prsm_float arr[], const size_t arr_size) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(arr != NULL, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(arr_size == prsm_tensor_size(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_SHAPES));

    // copy data
    vt_memcopy(t->data, arr, arr_size * sizeof(prsm_float));
}

void prsm_tensor_swap(prsm_tensor_t *const lhs, prsm_tensor_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(lhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(rhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // swap elements
    vt_pswap((void*)&lhs, (void*)&rhs);
}

/* 
    Tensor slicing/view operations
*/

bool prsm_tensor_is_view(const prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    return t->is_view;
}

prsm_tensor_t prsm_tensor_make_view(const prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // create view of the entire tensor
    prsm_tensor_t tview = *t;
    tview.is_view = true;

    return tview;
}

prsm_tensor_t prsm_tensor_make_view_mat(const prsm_tensor_t *const t, const size_t dim) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(t->ndim >= 2, "%s: Can make view only of a higher dimension tensor.\n", prsm_status_to_str(PRSM_STATUS_ERROR_IS_REQUIRED));
    VT_ENFORCE(
        dim < t->ndim, 
        "%s: %zu < %zu\n", 
        prsm_status_to_str(PRSM_STATUS_ERROR_OUT_OF_BOUNDS_ACCESS),
        dim,
        t->ndim
    );

    // calcuate view size (number of elements)
    size_t view_size = 1;
    VT_FOREACH(i, 1, t->ndim) {
        view_size *= t->shape[i];
    }

    // find view on data
    prsm_float *view_data = t->data + view_size * dim;

    // create view of a matrix
    prsm_tensor_t tview = {
        .ndim = t->ndim - 1,
        .shape = t->shape + 1,
        .data = view_data,
        .is_view = true
    };

    return tview;
}

prsm_tensor_t prsm_tensor_make_view_vec(const prsm_tensor_t *const t, const size_t row) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(t->ndim == 2, "%s: Can make view only of a 2D tensor.\n", prsm_status_to_str(PRSM_STATUS_ERROR_IS_REQUIRED));
    VT_ENFORCE(row < t->shape[0], "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_OUT_OF_BOUNDS_ACCESS));

    // create vector view
    prsm_tensor_t tview = {
        .ndim = 1,
        .shape = &t->shape[1],
        .data = t->data + row * t->shape[1],
        .is_view = true
    };

    return tview;
}

prsm_tensor_t prsm_tensor_make_view_range(const prsm_tensor_t *const t, const size_t range[]) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(range != NULL, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // check shapes for out-of-bounds access
    VT_FOREACH(i, 0, t->ndim) {
        VT_ENFORCE(
            range[i] <= range[t->ndim+i],
            "%s: %zu < %zu\n", 
            prsm_status_to_str(PRSM_STATUS_ERROR_OUT_OF_BOUNDS_ACCESS),
            range[i],
            range[t->ndim+i]
        );

        VT_ENFORCE(
            range[t->ndim+i] < t->shape[i],
            "%s: %zu < %zu\n", 
            prsm_status_to_str(PRSM_STATUS_ERROR_OUT_OF_BOUNDS_ACCESS),
            range[t->ndim+i],
            t->shape[i]
        );
    }

    // calculate view start and adjust shape
    size_t view_size = 1;
    size_t view_start = 1;
    VT_FOREACH(i, 0, t->ndim) {
        view_size *= range[t->ndim+i] - range[i] + 1;
        view_start *= range[i];
    }

    // create view
    prsm_tensor_t tview = {
        .ndim = 1,
        ._shape = view_size,
        .data = t->data + view_start,
        .is_view = true
    };
    tview.shape = &tview._shape;

    return tview;
}

/* 
    Tensor get/set value operations
*/

prsm_float prsm_tensor_get_val(const prsm_tensor_t *const t, const size_t idx) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    return t->data[idx];
}

void prsm_tensor_set_val(prsm_tensor_t *const t, const size_t idx, const prsm_float value) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    t->data[idx] = value;
}

void prsm_tensor_set_all(prsm_tensor_t *const t, const prsm_float value) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    VT_FOREACH(i, 0, prsm_tensor_size(t)) {
        t->data[i] = value;
    }
}

void prsm_tensor_set_ones(prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    prsm_tensor_set_all(t, 1);
}

void prsm_tensor_set_zeros(prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    prsm_tensor_set_all(t, 0);
}

void prsm_tensor_set_identity(prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    
    // set identity
    if (t->ndim == 1) {
        t->data[0] = 1;
    } else if (t->ndim == 2) {
        VT_FOREACH(i, 0, t->shape[0]) {
            VT_FOREACH(j, 0, t->shape[1]) {
                if (i == j) {
                    t->data[vt_index_2d_to_1d(i, j, t->shape[1])] = 1;
                }
            }
        }
    } else if (t->ndim == 3) {
        VT_FOREACH(i, 0, t->shape[0]) {
            VT_FOREACH(j, 0, t->shape[1]) {
                VT_FOREACH(k, 0, t->shape[2]) {
                    if (i == j) {
                        t->data[vt_index_3d_to_1d(i, j, k, t->shape[0], t->shape[1])] = 1;
                    }
                }
            }
        }
    } else {
        VT_ENFORCE(0, "%s: Higher dimensions are not supported!\n", prsm_status_to_str(PRSM_STATUS_OPERATION_FAILURE));
    }
}

/* 
    Tensor-wise operations
*/

prsm_float prsm_tensor_dot(const prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(lhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(rhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(lhs->ndim == 1 && rhs->ndim == 1, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_DIMENSIONS));

    // calculate result
    prsm_float result = 0;
    const size_t size = prsm_tensor_size(lhs);
    VT_FOREACH(i, 0, size) {
        result += lhs->data[i] * rhs->data[i];
    }

    return result;
}

prsm_tensor_t *prsm_tensor_sum(prsm_tensor_t *out, const prsm_tensor_t *const in, const uint8_t axis) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(in), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(in->ndim < 4, "%s: Higher dimensions are not supported!\n", prsm_status_to_str(PRSM_STATUS_OPERATION_FAILURE));

    // create tensor
    prsm_tensor_t *ret = (out == NULL)
        ? prsm_tensor_create_vec(in->alloctr, in->shape[0])
        : out;

    // switch dimension
    switch (in->ndim) { // TODO: check if ret needs resizing (out may be just the right size-shape)
        case 1:
            {   
                // resize
                if (!prsm_tensor_match_shape_ex(ret, 1, &(size_t[]) {1})) prsm_tensor_resize(ret, 1, 1);
                prsm_tensor_set_val(ret, 0, prsm_tensor_calc_sum(in));
            }
            break;
        case 2:
            {   
                // resize
                prsm_tensor_resize(ret, 1, in->shape[!axis]);
                
                // transpose if sum along rows
                if (!axis) prsm_tensor_transpose((prsm_tensor_t*)in);

                // sum up values
                const size_t size = prsm_tensor_size(ret);
                VT_FOREACH(i, 0, size) {
                    const prsm_tensor_t tmp = prsm_tensor_make_view_vec(in, i);
                    prsm_tensor_set_val(ret, i, prsm_tensor_calc_sum(&tmp));
                }

                // undo: transpose if sum along rows
                if (!axis) prsm_tensor_transpose((prsm_tensor_t*)in);
            }
            break;
            case 3:
            {   
                // z-axis
                if (axis == 0) {
                    // resize
                    prsm_tensor_resize(ret, 2, in->shape[1], in->shape[2]);

                    // zero-init
                    prsm_tensor_set_all(ret, 0);

                    // sum up values
                    const size_t ndim = in->shape[0];
                    VT_FOREACH(i, 0, ndim) {
                        const prsm_tensor_t tmp = prsm_tensor_make_view_mat(in, i);
                        prsm_tensor_add(ret, ret, &tmp);
                    }
                } else if (axis == 1) { // row-wise summation
                    // resize
                    prsm_tensor_resize(ret, 3, 1, in->shape[2]);
                    
                    // sum up values
                    const size_t ndim = in->shape[0];
                    VT_FOREACH(i, 0, ndim) {
                        const prsm_tensor_t tmp = prsm_tensor_make_view_mat(in, i);

                        // extract row where to save the result
                        prsm_tensor_t tmp_ret_mat = prsm_tensor_make_view_mat(ret, i);
                        prsm_tensor_t tmp_ret_vec = prsm_tensor_make_view_vec(&tmp_ret_mat, 0);

                        // calculate sum
                        prsm_tensor_sum(&tmp_ret_vec, &tmp, 0);
                    }
                } else { // col-wise summation
                    VT_UNIMPLEMENTED("TODO: col-wise summation for 3d matrix");
                }
            }
            break;
        default:
            break;
    }

    return ret;
}

prsm_tensor_t *prsm_tensor_add(prsm_tensor_t *out, const prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(lhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(rhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(prsm_tensor_match_shape(lhs, rhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_SHAPES));

    // create tensor
    prsm_tensor_t *ret = (out == NULL)
        ? prsm_tensor_create_shape(lhs->alloctr, lhs->ndim, lhs->shape)
        : out;

    // check size
    if (!prsm_tensor_match_shape(ret, lhs)) {
        prsm_tensor_resize_shape(ret, lhs->ndim, lhs->shape);
    }

    // add tensors
    const size_t size = prsm_tensor_size(ret);
    VT_FOREACH(i, 0, size) {
        ret->data[i] = lhs->data[i] + rhs->data[i];
    }

    return ret;
}

prsm_tensor_t *prsm_tensor_sub(prsm_tensor_t *out, const prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(lhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(rhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(prsm_tensor_match_shape(lhs, rhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_SHAPES));

    // create tensor
    prsm_tensor_t *ret = (out == NULL)
        ? prsm_tensor_create_shape(lhs->alloctr, lhs->ndim, lhs->shape)
        : out;

    // check size
    if (!prsm_tensor_match_shape(ret, lhs)) {
        prsm_tensor_resize_shape(ret, lhs->ndim, lhs->shape);
    }

    // add tensors
    const size_t size = prsm_tensor_size(ret);
    VT_FOREACH(i, 0, size) {
        ret->data[i] = lhs->data[i] - rhs->data[i];
    }

    return ret;
}

prsm_tensor_t *prsm_tensor_mul(prsm_tensor_t *out, const prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(lhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(rhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(lhs->ndim < 3 && rhs->ndim < 3, "%s: Higher dimensions are not supported!\n", prsm_status_to_str(PRSM_STATUS_OPERATION_FAILURE));
    VT_ENFORCE(!(lhs->ndim == 1 && rhs->ndim == 1), "%s: Use 'prsm_tensor_dot(lhs, rhs)' for vectors!\n", prsm_status_to_str(PRSM_STATUS_OPERATION_FAILURE));

    /*
     * v - vector
     * m - matrix (2d)
     * 
     * There are 3 cases:
     *  1. v * m
     *  2. m * v
     *  3. m * m
     */

    // calculate multiplication
    if (lhs->ndim == 1 && rhs->ndim == 2) {                   // case 2: v * m
        return prsm_tensor_mul_vec_by_mat(out, lhs, rhs);
    } else if (lhs->ndim == 2 && rhs->ndim == 1) {            // case 3: m * v
        return prsm_tensor_mul_mat_by_vec(out, lhs, rhs);
    } else { // if (lhs->ndim == 2 && rhs->ndim == 2) {       // case 4: m * m
        return prsm_tensor_mul_mat_by_mat(out, lhs, rhs);
    }
}

prsm_tensor_t *prsm_tensor_mul_elwise(prsm_tensor_t *out, const prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(lhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(rhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(prsm_tensor_match_shape(lhs, rhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_SHAPES));

    // create tensor
    prsm_tensor_t *ret = (out == NULL)
        ? prsm_tensor_create_shape(lhs->alloctr, lhs->ndim, lhs->shape)
        : out;

    // check size
    if (!prsm_tensor_match_shape(ret, lhs)) {
        prsm_tensor_resize_shape(ret, lhs->ndim, lhs->shape);
    }

    // perform element-wise multiplication
    const size_t size = prsm_tensor_size(ret);
    VT_FOREACH(i, 0, size) {
        ret->data[i] = lhs->data[i] * rhs->data[i];
    }

    return ret;
}

/* 
    Tensor element-wise operations
*/

void prsm_tensor_apply_scale_add(prsm_tensor_t *const t, const prsm_float sval, const prsm_float aval) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // scale and add
    const size_t size = prsm_tensor_size(t);
    VT_FOREACH(i, 0, size) {
        t->data[i] = t->data[i] * sval + aval;
    }
}

void prsm_tensor_apply_ceil(prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    const size_t size = prsm_tensor_size(t);
    VT_FOREACH(i, 0, size) {
        t->data[i] = PRSM_CEIL(t->data[i]);
    }
}

void prsm_tensor_apply_floor(prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    const size_t size = prsm_tensor_size(t);
    VT_FOREACH(i, 0, size) {
        t->data[i] = PRSM_FLOOR(t->data[i]);
    }
}

void prsm_tensor_apply_round(prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    const size_t size = prsm_tensor_size(t);
    VT_FOREACH(i, 0, size) {
        t->data[i] = PRSM_ROUND(t->data[i]);
    }
}

void prsm_tensor_apply_clip(prsm_tensor_t *const t, const prsm_float min, const prsm_float max) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    const size_t size = prsm_tensor_size(t);
    VT_FOREACH(i, 0, size) {
        t->data[i] = PRSM_CLAMP(t->data[i], min, max);
    }
}

void prsm_tensor_apply_abs(prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    const size_t size = prsm_tensor_size(t);
    VT_FOREACH(i, 0, size) {
        t->data[i] = PRSM_ABS(t->data[i]);
    }
}

void prsm_tensor_apply_neg(prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    const size_t size = prsm_tensor_size(t);
    VT_FOREACH(i, 0, size) {
        t->data[i] = -t->data[i];
    }
}

void prsm_tensor_apply_func(prsm_tensor_t *const t, prsm_float (*func)(prsm_float)) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(func != NULL, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    const size_t size = prsm_tensor_size(t);
    VT_FOREACH(i, 0, size) {
        t->data[i] = func(t->data[i]);
    }
}

/* 
    Tensor statistics on the whole tensor
*/

prsm_float prsm_tensor_get_min(const prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    prsm_float val = t->data[0];
    const size_t size = prsm_tensor_size(t);
    VT_FOREACH(i, 1, size) {
        if (val > t->data[i]) {
            val = t->data[i];
        }
    }

    return val;
}

prsm_float prsm_tensor_get_max(const prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    prsm_float val = t->data[0];
    const size_t size = prsm_tensor_size(t);
    VT_FOREACH(i, 1, size) {
        if (val < t->data[i]) {
            val = t->data[i];
        }
    }

    return val;
}

void prsm_tensor_get_minmax(const prsm_tensor_t *const t, prsm_float *min, prsm_float *max) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    *min = *max = t->data[0];
    const size_t size = prsm_tensor_size(t);
    VT_FOREACH(i, 1, size) {
        if (*min > t->data[i]) {
            *min = t->data[i];
        }

        if (*max < t->data[i]) {
            *max = t->data[i];
        }
    }
}

size_t prsm_tensor_get_min_index(const prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    size_t min_idx = 0;
    prsm_float val = t->data[0];
    const size_t size = prsm_tensor_size(t);
    VT_FOREACH(i, 1, size) {
        if (val > t->data[i]) {
            min_idx = i;
            val = t->data[i];
        }
    }

    return min_idx;
}

size_t prsm_tensor_get_max_index(const prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    size_t max_idx = 0;
    prsm_float val = t->data[0];
    const size_t size = prsm_tensor_size(t);
    VT_FOREACH(i, 1, size) {
        if (val < t->data[i]) {
            max_idx = i;
            val = t->data[i];
        }
    }

    return max_idx;
}

void prsm_tensor_get_minmax_index(const prsm_tensor_t *const t, size_t *min_index, size_t *max_index) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    prsm_float min = t->data[0], max = t->data[0];
    const size_t size = prsm_tensor_size(t);
    VT_FOREACH(i, 1, size) {
        if (min > t->data[i]) {
            min = t->data[i];
            *min_index = i;
        }

        if (max < t->data[i]) {
            max = t->data[i];
            *max_index = i;
        }
    }
}

prsm_float prsm_tensor_calc_sum(const prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    prsm_float sum = 0;
    const size_t size = prsm_tensor_size(t);
    VT_FOREACH(i, 0, size) {
        sum += t->data[i];
    }

    return sum;
}

prsm_float prsm_tensor_calc_prod(const prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    prsm_float prod = 0;
    const size_t size = prsm_tensor_size(t);
    VT_FOREACH(i, 0, size) {
        prod *= t->data[i];
    }

    return prod;
}

prsm_float prsm_tensor_calc_mean(const prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    prsm_float sum = prsm_tensor_calc_sum(t);
    const size_t size = prsm_tensor_size(t);

    return sum/size;
}

prsm_float prsm_tensor_calc_var(const prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    prsm_float sum = 0;
    prsm_float mean = prsm_tensor_calc_mean(t);
    const size_t size = prsm_tensor_size(t);
    VT_FOREACH(i, 0, size) {
        sum = PRSM_POW(t->data[i] - mean, 2);
    }

    return sum/size;
}

prsm_float prsm_tensor_calc_std(const prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    return PRSM_SQRT(prsm_tensor_calc_var(t));
}

/* 
    Tensor rand operations
*/

void prsm_tensor_rand(prsm_tensor_t *const t) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // randomize
    const size_t size = prsm_tensor_size(t);
    VT_FOREACH(i, 0, size) {
        t->data[i] = vt_math_random_f32(1);
    }
}

void prsm_tensor_rand_uniform(prsm_tensor_t *const t, const prsm_float lbound, const prsm_float ubound) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // randomize
    const size_t size = prsm_tensor_size(t);
    VT_FOREACH(i, 0, size) {
        t->data[i] = vt_math_random_f32_uniform(lbound, ubound);
    }
}

void prsm_tensor_rand_normal(prsm_tensor_t *const t, const prsm_float mu, const prsm_float std) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // randomize
    const size_t size = prsm_tensor_size(t);
    VT_FOREACH(i, 0, size) {
        t->data[i] = vt_math_random_f32_normal(mu, std);
    }
}

/* 
    Pretty printing
*/

void prsm_tensor_display(const prsm_tensor_t *const t, const size_t range[]) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // print data
    if (range == NULL && t->ndim == 1) {
        VT_FOREACH(i, 0, t->shape[0]) {
            printf("%s%.2f%s", (i == 0 ? "[ " : "  "), t->data[i], (i == t->shape[0] - 1 ? " ]\n": "\n"));
        }
    } else if (range == NULL && t->ndim == 2) {
        VT_FOREACH(i, 0, t->shape[0]) {
            printf("%s", i == 0 ? "[ [ " : "  [ ");
            VT_FOREACH(j, 0, t->shape[1]) {
                printf("%-*.2f ", j == t->shape[1]-1 ? 0 : 7, t->data[vt_index_2d_to_1d(i, j, t->shape[1])]);
            }
            printf("%s", i == t->shape[0]-1 ? "] ]\n" : "]\n");
        }
    } else if (range == NULL && t->ndim == 3) {
        VT_FOREACH(k, 0, t->shape[0]) {
            printf("%s", k == 0 ? "[ " : "  ");
            VT_FOREACH(i, 0, t->shape[1]) {
                printf("%s", i == 0 ? "[ [ " : "    [ ");
                VT_FOREACH(j, 0, t->shape[2]) {
                    printf("%-*.2f ", j == t->shape[2]-1 ? 0 : 7, t->data[vt_index_3d_to_1d(i, j, k, t->shape[1], t->shape[2])]);
                }
                if (k != t->shape[0]-1) printf("%s", i == t->shape[1]-1 ? "] ]\n" : "]\n");
                else printf("%s", i == t->shape[1]-1 ? "] ] ]\n" : "]\n");
            }
        }
    } else {
        VT_UNIMPLEMENTED("Unsupported for now!");
    }

    // print shape
    VT_FOREACH(i, 0, t->ndim) {
        printf("%s%zu%s", (i == 0 ? "Shape: (": ", "), t->shape[i], (i == t->ndim-1 ? ")\n" : ""));
    }
}

// -------------------------- PRIVATE -------------------------- //

/**
 * @brief  Multiplies vector by matrix
 * @param  out output tensor
 * @param  lhs input vector tensor
 * @param  rhs input matrix tensor
 * @returns vector tensor
 * 
 * @note if `out==NULL`, tensor is allocated
 */
static prsm_tensor_t *prsm_tensor_mul_vec_by_mat(prsm_tensor_t *const out, const prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(lhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(rhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(lhs->ndim == 1 && rhs->ndim == 2, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_DIMENSIONS));
    VT_ENFORCE(lhs->shape[0] == rhs->shape[0], "%s: Maybe transpose the matrix.\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_SHAPES));

    // create tensor
    const size_t size = rhs->shape[1];
    prsm_tensor_t *tresult = (out == NULL) 
        ? prsm_tensor_create(lhs->alloctr, 1, size)
        : out;

    // check size
    if (tresult->ndim != 1 || prsm_tensor_size(tresult) != size) {
        prsm_tensor_resize(tresult, 1, size);
    }

    // zero out the values
    prsm_tensor_set_all(tresult, 0);

    // calculate multiplication
    const size_t lhs_size = prsm_tensor_size(lhs);
    VT_FOREACH(i, 0, size) {
        VT_FOREACH(j, 0, lhs_size) {
            tresult->data[i] += lhs->data[j] * rhs->data[vt_index_2d_to_1d(j, i, size)];
        }
    }

    return tresult;
}

/**
 * @brief  Multiplies matrix by vector
 * @param  out output tensor
 * @param  lhs input matrix tensor
 * @param  rhs input vector tensor
 * @returns matrix tensor
 * 
 * @note if `out==NULL`, tensor is allocated
 */
static prsm_tensor_t *prsm_tensor_mul_mat_by_vec(prsm_tensor_t *const out, const prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(lhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(rhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(lhs->ndim == 2 && rhs->ndim == 1, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_DIMENSIONS));
    VT_ENFORCE(lhs->shape[1] == rhs->shape[0], "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_SHAPES));

    // create tensor
    const size_t size = lhs->shape[0];
    prsm_tensor_t *tresult = (out == NULL) 
        ? prsm_tensor_create(lhs->alloctr, 1, size)
        : out;

    // check size
    if (tresult->ndim != 1 || prsm_tensor_size(tresult) != size) {
        prsm_tensor_resize(tresult, 1, size);
    }

    // zero out the values
    prsm_tensor_set_all(tresult, 0);

    // calculate multiplication
    const size_t rhs_size = prsm_tensor_size(rhs);
    VT_FOREACH(i, 0, size) {
        VT_FOREACH(j, 0, rhs_size) {
            tresult->data[i] += lhs->data[vt_index_2d_to_1d(i, j, rhs_size)] * rhs->data[j];
        }
    }

    return tresult;
}

/**
 * @brief  Multiplies matrix by matrix
 * @param  out output tensor
 * @param  lhs input vector tensor
 * @param  rhs input vector tensor
 * @returns matrix tensor
 * 
 * @note if `out==NULL`, tensor is allocated
 */
static prsm_tensor_t *prsm_tensor_mul_mat_by_mat(prsm_tensor_t *const out, const prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(lhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(rhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(lhs->ndim == 2 && rhs->ndim == 2, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_DIMENSIONS));
    VT_ENFORCE(lhs->shape[1] == rhs->shape[0], "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_SHAPES));

    // create tensor
    const size_t rows = lhs->shape[0];
    const size_t cols = rhs->shape[1];
    prsm_tensor_t *tresult = (out == NULL) 
        ? prsm_tensor_create(lhs->alloctr, 2, rows, cols)
        : out;

    // check size
    if (tresult->ndim != 2 || prsm_tensor_size(tresult) != rows * cols) {
        prsm_tensor_resize(tresult, 2, rows, cols);
    }

    // zero out the values
    prsm_tensor_set_all(tresult, 0);

    // calculate multiplication
    VT_FOREACH(i, 0, rows) {
        VT_FOREACH(j, 0, cols) {
            VT_FOREACH(k, 0, rhs->shape[0]) {
                tresult->data[vt_index_2d_to_1d(i, j, cols)] += lhs->data[vt_index_2d_to_1d(i, k, cols)] * rhs->data[vt_index_2d_to_1d(k, j, cols)];
            }
        }
    }

    return tresult;
}


