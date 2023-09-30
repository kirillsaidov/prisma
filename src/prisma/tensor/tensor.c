#include "prisma/tensor/tensor.h"

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
    return (
        !prsm_tensor_is_null(t1) &&
        !prsm_tensor_is_null(t2) &&
        vt_memcmp(t1, t2, sizeof(prsm_tensor_t))
    );
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

bool prsm_tensor_is_view(const prsm_tensor_t *const t);
prsm_tensor_t prsm_tensor_make_view(const prsm_tensor_t *const t);
prsm_tensor_t prsm_tensor_make_view_mat(const prsm_tensor_t *const t, const size_t dim);
prsm_tensor_t prsm_tensor_make_view_vec(const prsm_tensor_t *const t, const size_t dim, const size_t col);
prsm_tensor_t prsm_tensor_make_view_array(const prsm_tensor_t *const t, const size_t dim, const size_t row);
prsm_tensor_t prsm_tensor_make_view_range(const prsm_tensor_t *const t, const size_t *const shapeFrom, const size_t *const shapeTo);

/* 
    Tensor get/set value operations
*/

prsm_float prsm_tensor_get_v(const prsm_tensor_t *const t, const size_t idx);
void prsm_tensor_set_v(prsm_tensor_t *const t, const size_t idx, const prsm_float value);
void prsm_tensor_set_all(prsm_tensor_t *const t, const prsm_float value);
void prsm_tensor_set_ones(prsm_tensor_t *const t);
void prsm_tensor_set_zeros(prsm_tensor_t *const t);
void prsm_tensor_set_identity(prsm_tensor_t *const t);

/* 
    Tensor-wise operations
*/

prsm_tensor_t *prsm_tensor_add(const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);
prsm_tensor_t *prsm_tensor_sub(const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);
prsm_tensor_t *prsm_tensor_mul(const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);
prsm_tensor_t *prsm_tensor_div(const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);

enum PrismaStatus prsm_tensor_add_into(prsm_tensor_t *const tout, const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);
enum PrismaStatus prsm_tensor_sub_into(prsm_tensor_t *const tout, const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);
enum PrismaStatus prsm_tensor_mul_into(prsm_tensor_t *const tout, const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);
enum PrismaStatus prsm_tensor_div_into(prsm_tensor_t *const tout, const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);

/* 
    Tensor element-wise operations
*/

void prsm_tensor_apply_scale_add(prsm_tensor_t *const t, const prsm_float sval, const prsm_float aval);
void prsm_tensor_apply_ceil(prsm_tensor_t *const t);
void prsm_tensor_apply_floor(prsm_tensor_t *const t);
void prsm_tensor_apply_round(prsm_tensor_t *const t);
void prsm_tensor_apply_clip(prsm_tensor_t *const t, const prsm_float min, const prsm_float max);
void prsm_tensor_apply_abs(prsm_tensor_t *const t);
void prsm_tensor_apply_neg(prsm_tensor_t *const t);

/* 
    Tensor statistics on the whole tensor
*/

prsm_float prsm_tensor_get_min(const prsm_tensor_t *const t);
prsm_float prsm_tensor_get_max(const prsm_tensor_t *const t);
void prsm_tensor_get_minmax(const prsm_tensor_t *const t, prsm_float *min, prsm_float *max);
size_t prsm_tensor_get_min_index(const prsm_tensor_t *const t);
size_t prsm_tensor_get_max_index(const prsm_tensor_t *const t);
void prsm_tensor_get_minmax_index(const prsm_tensor_t *const t, size_t *min_index, size_t *max_index);

prsm_float prsm_tensor_calc_sum(const prsm_tensor_t *const t);
prsm_float prsm_tensor_calc_prod(const prsm_tensor_t *const t);
prsm_float prsm_tensor_calc_mean(const prsm_tensor_t *const t);
prsm_float prsm_tensor_calc_var(const prsm_tensor_t *const t);
prsm_float prsm_tensor_calc_stddev(const prsm_tensor_t *const t);

/* 
    Tensor rand operations
*/

void prsm_tensor_rand(prsm_tensor_t *const t);
void prsm_tensor_rand_uniform(prsm_tensor_t *const t, const prsm_float lbound, const prsm_float ubound);
void prsm_tensor_rand_normal(prsm_tensor_t *const t, const prsm_float mu, const prsm_float sigma);
void prsm_tensor_rand_std_normal(prsm_tensor_t *const t);

