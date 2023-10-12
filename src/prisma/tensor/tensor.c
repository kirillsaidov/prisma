#include "prisma/tensor/tensor.h"

static prsm_tensor_t *prsm_tensor_mul_vec_by_mat(prsm_tensor_t *const tout, const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);
static prsm_tensor_t *prsm_tensor_mul_mat_by_vec(prsm_tensor_t *const tout, const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);
static prsm_tensor_t *prsm_tensor_mul_mat_by_mat(prsm_tensor_t *const tout, const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);

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

prsm_tensor_t *prsm_tensor_create_shape(struct VitaBaseAllocatorType *const alloctr, const size_t ndim, const size_t *const shape) {
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

    // find new shape
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

void prsm_tensor_resize_shape(prsm_tensor_t *const t, const size_t ndim, const size_t *const shape)  {
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

    // find new shape
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

void prsm_tensor_dup_into(prsm_tensor_t *const tout, const prsm_tensor_t *const tin) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(tout), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(tin), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(prsm_tensor_match_shape(tout, tin), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_SHAPES));
    
    // copy data
    vt_memcopy(tout->data, tin->data, prsm_tensor_size(tout) * sizeof(prsm_float));
}

/* 
    Tensor data operations
*/

bool prsm_tensor_match_shape(const prsm_tensor_t *const t1, const prsm_tensor_t *const t2) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t1), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t2), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(t1->ndim == t2->ndim, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_DIMENSIONS));

    return vt_memcmp(t1->shape, t2->shape, t1->ndim * sizeof(size_t));
}

bool prsm_tensor_equals(const prsm_tensor_t *const t1, const prsm_tensor_t *const t2) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t1), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t2), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    if (t1->ndim != t2->ndim) return false;
    if (prsm_tensor_size(t1) != prsm_tensor_size(t2)) return false;

    // check values
    VT_FOREACH(i, 0, prsm_tensor_size(t1)) {
        if (t1->data[i] != t2->data[i]) return false;
    }

    return true;
}

void prsm_tensor_assign(prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(lhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(rhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(prsm_tensor_match_shape(lhs, rhs), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_SHAPES));

    // copy data
    vt_memcopy(lhs->data, rhs->data, prsm_tensor_size(lhs) * sizeof(prsm_float));
}

void prsm_tensor_swap(prsm_tensor_t *const t1, prsm_tensor_t *const t2) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t1), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t2), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // swap elements
    vt_pswap((void*)&t1, (void*)&t2);
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

prsm_tensor_t prsm_tensor_make_view_vec(const prsm_tensor_t *const t, const size_t idxFrom, const size_t idxTo) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(idxFrom < idxTo && idxTo < prsm_tensor_size(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_OUT_OF_BOUNDS_ACCESS));

    // create vector view
    prsm_tensor_t tview = {
        .ndim = 1,
        .size = idxTo - idxFrom,
        .data = t->data + idxFrom,
        .is_view = true
    };

    return tview;
}

prsm_tensor_t prsm_tensor_make_view_range(const prsm_tensor_t *const t, const size_t *const shapeFrom, const size_t *const shapeTo) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(shapeFrom != NULL, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(shapeTo != NULL, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // check shapes for out-of-bounds access
    VT_FOREACH(i, 0, t->ndim) {
        VT_ENFORCE(
            shapeFrom[i] <= shapeTo[i],
            "%s: %zu < %zu\n", 
            prsm_status_to_str(PRSM_STATUS_ERROR_OUT_OF_BOUNDS_ACCESS),
            shapeFrom[i],
            shapeTo[i]
        );

        VT_ENFORCE(
            shapeTo[i] < t->shape[i],
            "%s: %zu < %zu\n", 
            prsm_status_to_str(PRSM_STATUS_ERROR_OUT_OF_BOUNDS_ACCESS),
            shapeTo[i],
            t->shape[i]
        );
    }

    // calculate view start and adjust shape
    size_t view_size = 1;
    size_t view_start = 1;
    VT_FOREACH(i, 0, t->ndim) {
        view_size *= shapeTo[i] - shapeFrom[i] + 1;
        view_start *= shapeFrom[i];
    }

    // create view
    prsm_tensor_t tview = {
        .ndim = 1,
        .size = view_size,
        .data = t->data + view_start,
        .is_view = true
    };

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

prsm_float prsm_tensor_dot(const prsm_tensor_t *const t1, const prsm_tensor_t *const t2) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t1), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t2), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(t1->ndim == 1 && t2->ndim == 1, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_DIMENSIONS));

    // calculate result
    prsm_float result = 0;
    const size_t size = prsm_tensor_size(t1);
    VT_FOREACH(i, 0, size) {
        result += t1->data[i] * t2->data[i];
    }

    return result;
}

prsm_tensor_t *prsm_tensor_add(prsm_tensor_t *tout, const prsm_tensor_t *const t1, const prsm_tensor_t *const t2) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t1), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t2), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(prsm_tensor_match_shape(t1, t2), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_SHAPES));

    // create tensor
    prsm_tensor_t *tret = (tout == NULL)
        ? prsm_tensor_create_shape(t1->alloctr, t1->ndim, t1->shape)
        : tout;

    // check size
    if (!prsm_tensor_match_shape(tret, t1)) {
        prsm_tensor_resize_shape(tret, t1->ndim, t1->shape);
    }

    // zero out the values
    prsm_tensor_set_all(tret, 0);

    // add tensors
    const size_t size = prsm_tensor_size(tret);
    VT_FOREACH(i, 0, size) {
        tret->data[i] = t1->data[i] + t2->data[i];
    }

    return tret;
}

prsm_tensor_t *prsm_tensor_sub(prsm_tensor_t *tout, const prsm_tensor_t *const t1, const prsm_tensor_t *const t2) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t1), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t2), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(prsm_tensor_match_shape(t1, t2), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_SHAPES));

    // create tensor
    prsm_tensor_t *tret = (tout == NULL)
        ? prsm_tensor_create_shape(t1->alloctr, t1->ndim, t1->shape)
        : tout;

    // check size
    if (!prsm_tensor_match_shape(tret, t1)) {
        prsm_tensor_resize_shape(tret, t1->ndim, t1->shape);
    }

    // zero out the values
    prsm_tensor_set_all(tret, 0);

    // add tensors
    const size_t size = prsm_tensor_size(tret);
    VT_FOREACH(i, 0, size) {
        tret->data[i] = t1->data[i] - t2->data[i];
    }

    return tret;
}

prsm_tensor_t *prsm_tensor_mul(prsm_tensor_t *tout, const prsm_tensor_t *const t1, const prsm_tensor_t *const t2) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t1), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t2), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(t1->ndim < 3 && t2->ndim < 3, "%s: Higher dimensions are not supported!\n", prsm_status_to_str(PRSM_STATUS_OPERATION_FAILURE));
    VT_ENFORCE(t1->ndim != 1 && t2->ndim != 1, "%s: Use 'prsm_tensor_dot(t1, t2)'!\n", prsm_status_to_str(PRSM_STATUS_OPERATION_FAILURE));

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
    if (t1->ndim == 1 && t2->ndim == 2) {                   // case 2: v * m
        return prsm_tensor_mul_vec_by_mat(tout, t1, t2);
    } else if (t1->ndim == 2 && t2->ndim == 1) {            // case 3: m * v
        return prsm_tensor_mul_mat_by_vec(tout, t1, t2);
    } else { // if (t1->ndim == 2 && t2->ndim == 2) {       // case 4: m * m
        return prsm_tensor_mul_mat_by_mat(tout, t1, t2);
    }
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

/* 
    Pretty printing
*/

void prsm_tensor_display(const prsm_tensor_t *const t, const size_t range[]) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // print
    if (range == NULL && t->ndim == 1) {
        printf("[ %.2f\n", t->data[0]);
        VT_FOREACH(i, 1, t->shape[0]-1) {
            printf("  %.2f\n", t->data[i]);
        }
        printf("  %.2f ]\n", t->data[t->shape[0]-1]);
    } else if (range == NULL && t->ndim == 2) {
        VT_FOREACH(i, 0, t->shape[0]) {
            printf("%s", i == 0 ? "[ [ " : "  [ ");
            VT_FOREACH(j, 0, t->shape[1]) {
                printf("%.2f ", t->data[vt_index_2d_to_1d(i, j, t->shape[1])]);
            }
            printf("%s", i == t->shape[0]-1 ? "] ]\n" : "]\n");
        }
    } else {
        VT_UNIMPLEMENTED("Unsupported for now!");
    }
}

// -------------------------- PRIVATE -------------------------- //

/**
 * @brief  Multiplies vector by matrix
 * @param  tout output tensor
 * @param  t1 input vector tensor
 * @param  t2 input matrix tensor
 * @returns vector tensor
 * 
 * @note if `tout==NULL`, tensor is allocated
 */
static prsm_tensor_t *prsm_tensor_mul_vec_by_mat(prsm_tensor_t *const tout, const prsm_tensor_t *const t1, const prsm_tensor_t *const t2) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t1), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t2), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(t1->ndim == 1 && t2->ndim == 2, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_DIMENSIONS));
    VT_ENFORCE(t1->shape[0] == t2->shape[1], "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_SHAPES));

    // create tensor
    const size_t size = t2->shape[1];
    prsm_tensor_t *tresult = (tout == NULL) 
        ? prsm_tensor_create(t1->alloctr, 1, size)
        : tout;

    // check size
    if (tresult->ndim != 1 || prsm_tensor_size(tresult) != size) {
        prsm_tensor_resize(tresult, 1, size);
    }

    // zero out the values
    prsm_tensor_set_all(tresult, 0);

    // calculate multiplication
    VT_FOREACH(i, 0, size) {
        VT_FOREACH(j, 0, size) {
            tresult->data[i] = t1->data[i] * t2->data[vt_index_2d_to_1d(i, j, size)];
        }
    }

    return tresult;
}

/**
 * @brief  Multiplies matrix by vector
 * @param  tout output tensor
 * @param  t1 input matrix tensor
 * @param  t2 input vector tensor
 * @returns matrix tensor
 * 
 * @note if `tout==NULL`, tensor is allocated
 */
static prsm_tensor_t *prsm_tensor_mul_mat_by_vec(prsm_tensor_t *const tout, const prsm_tensor_t *const t1, const prsm_tensor_t *const t2) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t1), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t2), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(t1->ndim == 2 && t2->ndim == 1, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_DIMENSIONS));
    VT_ENFORCE(t1->shape[1] == t2->shape[0], "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_SHAPES));

    // create tensor
    const size_t size = t2->shape[0];
    prsm_tensor_t *tresult = (tout == NULL) 
        ? prsm_tensor_create(t1->alloctr, 1, size)
        : tout;

    // check size
    if (tresult->ndim != 1 || prsm_tensor_size(tresult) != size) {
        prsm_tensor_resize(tresult, 1, size);
    }

    // zero out the values
    prsm_tensor_set_all(tresult, 0);

    // calculate multiplication
    VT_FOREACH(i, 0, size) {
        VT_FOREACH(j, 0, size) {
            tresult->data[i] = t1->data[vt_index_2d_to_1d(i, j, size)] * t2->data[j];
        }
    }

    return tresult;
}

/**
 * @brief  Multiplies matrix by matrix
 * @param  tout output tensor
 * @param  t1 input vector tensor
 * @param  t2 input vector tensor
 * @returns matrix tensor
 * 
 * @note if `tout==NULL`, tensor is allocated
 */
static prsm_tensor_t *prsm_tensor_mul_mat_by_mat(prsm_tensor_t *const tout, const prsm_tensor_t *const t1, const prsm_tensor_t *const t2) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t1), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(t2), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(t1->ndim == 2 && t2->ndim == 2, "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_DIMENSIONS));
    VT_ENFORCE(t1->shape[1] == t2->shape[0], "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_SHAPES));

    // create tensor
    const size_t rows = t1->shape[0];
    const size_t cols = t2->shape[1];
    prsm_tensor_t *tresult = (tout == NULL) 
        ? prsm_tensor_create(t1->alloctr, 2, rows, cols)
        : tout;

    // check size
    if (tresult->ndim != 2 || prsm_tensor_size(tresult) != rows * cols) {
        prsm_tensor_resize(tresult, 2, rows, cols);
    }

    // zero out the values
    prsm_tensor_set_all(tresult, 0);

    // calculate multiplication
    VT_FOREACH(i, 0, rows) {
        VT_FOREACH(j, 0, cols) {
            VT_FOREACH(k, 0, t2->shape[0]) {
                tresult->data[vt_index_2d_to_1d(i, j, cols)] += t1->data[vt_index_2d_to_1d(i, k, cols)] * t2->data[vt_index_2d_to_1d(k, j, cols)];
            }
        }
    }

    return tresult;
}


