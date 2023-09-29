#ifndef PRISMA_CORE_TENSOR_H
#define PRISMA_CORE_TENSOR_H

/** TENSOR MODULE
    - prsm_tensor_create    
    - prsm_tensor_destroy    
*/

#include "prisma/core/core.h"
#include "vita/allocator/mallocator.h"

typedef struct PrismaTensor {
    bool is_view;       // defines if tensor is modifiable or only viewable

    size_t ndim;        // number or dimensions: 1d, 2d, 3d, nd.
    size_t *shape;      // tensor shape
    prsm_float *data;   // data ptr

    // allocator: if `NULL`, then calloc/realloc/free is used
    struct VitaBaseAllocatorType *alloctr;
} prsm_tensor_t;

/* 
    Tensor creation/destruction
*/

extern prsm_tensor_t *prsm_tensor_create(struct VitaBaseAllocatorType *const alloctr, const size_t ndim, ...);
extern prsm_tensor_t *prsm_tensor_create_shape(struct VitaBaseAllocatorType *const alloctr, const size_t ndim, const size_t *const shape);
extern prsm_tensor_t *prsm_tensor_create_vec(struct VitaBaseAllocatorType *const alloctr, const size_t len);
extern prsm_tensor_t *prsm_tensor_create_mat(struct VitaBaseAllocatorType *const alloctr, const size_t rows, const size_t cols);
extern void prsm_tensor_destroy(prsm_tensor_t *t);

/* 
    Tensor data structure properties
*/

extern size_t prsm_tensor_dim(const prsm_tensor_t *const t);
extern const size_t *prsm_tensor_shape(const prsm_tensor_t *const t);
extern prsm_float *prsm_tensor_data(const prsm_tensor_t *const t);

/* 
    Tensor data structure operations
*/

extern bool prsm_tensor_resize(prsm_tensor_t *const t);
extern prsm_tensor_t *prsm_tensor_dup(const prsm_tensor_t *const t);
extern bool prsm_tensor_dup_into(prsm_tensor_t *const tout, const prsm_tensor_t *const t);

/* 
    Tensor slicing/view operations
*/

// TODO: ditto

/* 
    Tensor operations on elements
*/

extern float prsm_tensor_get(const prsm_tensor_t *const t, const size_t idx);
extern void prsm_tensor_set(prsm_tensor_t *const t, const size_t idx, const prsm_float value);
extern void prsm_tensor_set_all(prsm_tensor_t *const t, const prsm_float value);
extern void prsm_tensor_set_ones(prsm_tensor_t *const t);
extern void prsm_tensor_set_zeros(prsm_tensor_t *const t);

/* 
    Tensor data operations
*/

extern bool prsm_tensor_is_null(const prsm_tensor_t *const t);
extern bool prsm_tensor_equals(const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);
extern bool prsm_tensor_assign(prsm_tensor_t *const t1, const prsm_tensor_t *const t2);
extern bool prsm_tensor_swap(prsm_tensor_t *const t1, prsm_tensor_t *const t2);

extern void prsm_tensor_abs(prsm_tensor_t *const t);
extern void prsm_tensor_neg(prsm_tensor_t *const t);

/* 
    Tensor-wise operations
*/

extern prsm_tensor_t *prsm_tensor_add(const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);
extern prsm_tensor_t *prsm_tensor_sub(const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);
extern prsm_tensor_t *prsm_tensor_mul(const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);
extern prsm_tensor_t *prsm_tensor_div(const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);

extern bool prsm_tensor_add_into(prsm_tensor_t *const tout, const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);
extern bool prsm_tensor_sub_into(prsm_tensor_t *const tout, const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);
extern bool prsm_tensor_mul_into(prsm_tensor_t *const tout, const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);
extern bool prsm_tensor_div_into(prsm_tensor_t *const tout, const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);

extern void prsm_tensor_scale_add(prsm_tensor_t *const t, const prsm_float sval, const prsm_float aval);
extern void prsm_tensor_apply_ceil(prsm_tensor_t *const t);
extern void prsm_tensor_apply_floor(prsm_tensor_t *const t);
extern void prsm_tensor_apply_round(prsm_tensor_t *const t);
extern void prsm_tensor_apply_clip(prsm_tensor_t *const t, const prsm_float min, const prsm_float max);

/* 
    Tensor statistics on the whole tensor
*/

extern prsm_float prsm_tensor_get_min(const prsm_tensor_t *const t);
extern prsm_float prsm_tensor_get_max(const prsm_tensor_t *const t);
extern void prsm_tensor_get_minmax(const prsm_tensor_t *const t, prsm_float *min, prsm_float *max);
extern size_t prsm_tensor_get_min_index(const prsm_tensor_t *const t);
extern size_t prsm_tensor_get_max_index(const prsm_tensor_t *const t);
extern void prsm_tensor_get_minmax_index(const prsm_tensor_t *const t, size_t *min_index, size_t *max_index);

extern prsm_float prsm_tensor_calc_sum(const prsm_tensor_t *const t);
extern prsm_float prsm_tensor_calc_prod(const prsm_tensor_t *const t);
extern prsm_float prsm_tensor_calc_mean(const prsm_tensor_t *const t);
extern prsm_float prsm_tensor_calc_var(const prsm_tensor_t *const t);
extern prsm_float prsm_tensor_calc_stddev(const prsm_tensor_t *const t);

/* 
    Tensor statistics dim/stride(row:width)/size(col:height) wise
*/

// TODO: ditto

/* 
    Tensor rand operations
*/

extern void prsm_tensor_rand(prsm_tensor_t *const t);
extern void prsm_tensor_rand_uniform(prsm_tensor_t *const t, const prsm_float lbound, const prsm_float ubound);
extern void prsm_tensor_rand_normal(prsm_tensor_t *const t, const prsm_float mu, const prsm_float sigma);
extern void prsm_tensor_rand_std_normal(prsm_tensor_t *const t);

#endif // PRISMA_CORE_TENSOR_H

