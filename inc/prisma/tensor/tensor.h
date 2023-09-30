#ifndef PRISMA_CORE_TENSOR_H
#define PRISMA_CORE_TENSOR_H

/** TENSOR MODULE
    - prsm_tensor_create    
    - prsm_tensor_destroy    
*/

#include "prisma/core/core.h"
#include "vita/util/debug.h"
#include "vita/container/common.h"
#include "vita/allocator/mallocator.h"

typedef struct PrismaTensor {
    bool is_view;       // defines if tensor is modifiable or only viewable

    size_t ndim;        // number or dimensions: 1d, 2d, 3d, nd.
    union {
        size_t size;    // number of elements
        size_t *shape;  // tensor shape
    };
    prsm_float *data;   // data ptr

    // allocator: if `NULL`, then calloc/realloc/free is used
    struct VitaBaseAllocatorType *alloctr;
} prsm_tensor_t;

/* 
    Tensor creation/destruction
*/

/**
 * @brief  Creates a tensor
 * @param  alloctr allocator instance
 * @param  ndim number of dimensions
 * @param  ... tensor shape
 * @returns valid `prsm_tensor_t*` or asserts on failure
 */
extern prsm_tensor_t *prsm_tensor_create(struct VitaBaseAllocatorType *const alloctr, const size_t ndim, ...);

/**
 * @brief  Creates a tensor with from custom shape
 * @param  alloctr allocator instance
 * @param  ndim number of dimensions
 * @param  shape tensor shape
 * @returns valid `prsm_tensor_t*` or asserts on failure
 */
extern prsm_tensor_t *prsm_tensor_create_shape(struct VitaBaseAllocatorType *const alloctr, const size_t ndim, const size_t *const shape);

/**
 * @brief  Creates a tensor vector
 * @param  alloctr allocator instance
 * @param  len vector length
 * @returns valid `prsm_tensor_t*` or asserts on failure
 */
extern prsm_tensor_t *prsm_tensor_create_vec(struct VitaBaseAllocatorType *const alloctr, const size_t len);

/**
 * @brief  Creates a tensor matrix
 * @param  alloctr allocator instance
 * @param  rows number of rows
 * @param  cols number of cols
 * @returns valid `prsm_tensor_t*` or asserts on failure
 */
extern prsm_tensor_t *prsm_tensor_create_mat(struct VitaBaseAllocatorType *const alloctr, const size_t rows, const size_t cols);

/**
 * @brief  Creates a tensor matrix
 * @param  t tensor
 * @returns None
 */
extern void prsm_tensor_destroy(prsm_tensor_t *t);

/* 
    Tensor properties
*/

/**
 * @brief  Checks if tensor is valid
 * @param  t tensor
 * @returns ditto
 */
extern bool prsm_tensor_is_null(const prsm_tensor_t *const t);

/**
 * @brief  Returns number of tensor dimensions
 * @param  t tensor
 * @returns size_t ndim
 */
extern size_t prsm_tensor_dim(const prsm_tensor_t *const t);

/**
 * @brief  Returns tensor shape { x, y, z, ...}
 * @param  t tensor
 * @returns size_t shape[]
 */
extern const size_t *prsm_tensor_shape(const prsm_tensor_t *const t);

/**
 * @brief  Returns tensor data
 * @param  t tensor
 * @returns size_t *data
 */
extern prsm_float *prsm_tensor_data(const prsm_tensor_t *const t);

/**
 * @brief  Returns tensor size
 * @param  t tensor
 * @returns number of elements
 */
extern size_t prsm_tensor_size(const prsm_tensor_t *const t);

/* 
    Tensor data structure operations
*/

/**
 * @brief  Resizes tensor
 * @param  t tensor
 * @param  ndim number of dimensions
 * @param  ... tensor shape
 * @returns None
 */
extern void prsm_tensor_resize(prsm_tensor_t *const t, const size_t ndim, ...);

/**
 * @brief  Resizes tensor from custom shape
 * @param  t tensor
 * @param  ndim number of dimensions
 * @param  shape tensor shape
 * @returns None
 */
extern void prsm_tensor_resize_shape(prsm_tensor_t *const t, const size_t ndim, const size_t *const shape);

/**
 * @brief  Duplicates tensor
 * @param  t tensor
 * @returns tensor copy
 */
extern prsm_tensor_t *prsm_tensor_dup(const prsm_tensor_t *const t);

/**
 * @brief  Duplicates tensor into the preallocated tensor instance
 * @param  tout copy to tensor (output)
 * @param  tin copy from tensor (input)
 * @returns None
 */
extern void prsm_tensor_dup_into(prsm_tensor_t *const tout, const prsm_tensor_t *const tin);

/* 
    Tensor data operations
*/

/**
 * @brief  Checks if shapes match
 * @param  t1 tensor
 * @param  t2 tensor
 * @returns true if `t1.shape==t2.shape`
 */
extern bool prsm_tensor_match_shape(const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);

/**
 * @brief  Checks if tensors are equal
 * @param  t1 tensor
 * @param  t2 tensor
 * @returns true if `t1==t2`
 */
extern bool prsm_tensor_equals(const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);

/**
 * @brief  Assigns values to `lhs` from `rhs`
 * @param  lhs tensor
 * @param  rhs tensor
 * @returns None
 */
extern void prsm_tensor_assign(prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs);

/**
 * @brief  Swaps tensors
 * @param  t1 tensor
 * @param  t2 tensor
 * @returns None
 */
extern void prsm_tensor_swap(prsm_tensor_t *const t1, prsm_tensor_t *const t2);

/* 
    Tensor view: no allocations involved
*/

/**
 * @brief  Checks if tensor is a view object
 * @param  t tensor
 * @returns true if t.is_view
 */
extern bool prsm_tensor_is_view(const prsm_tensor_t *const t);

/**
 * @brief  Makes a view object from tensor
 * @param  t tensor
 * @returns prsm_tensor_t
 * 
 * @note it's a value type, no need to free it
 */
extern prsm_tensor_t prsm_tensor_make_view(const prsm_tensor_t *const t);

/**
 * @brief  Makes a view matrix from tensor
 * @param  t tensor
 * @param  dim view dimension
 * @returns prsm_tensor_t
 * 
 * @note it's a value type, no need to free it
 */
extern prsm_tensor_t prsm_tensor_make_view_mat(const prsm_tensor_t *const t, const size_t dim);

/**
 * @brief  Makes a view vector from tensor
 * @param  t tensor
 * @param  idxFrom start from index
 * @param  idxTo end index
 * @returns prsm_tensor_t
 * 
 * @note it's a value type, no need to free it
 */
extern prsm_tensor_t prsm_tensor_make_view_vec(const prsm_tensor_t *const t, const size_t idxFrom, const size_t idxTo);

/**
 * @brief  Makes a range view from tensor
 * @param  t tensor
 * @param  shapeFrom start shape range
 * @param  shapeTo end shape range
 * @returns prsm_tensor_t
 * 
 * @note it's a value type, no need to free it
 */
extern prsm_tensor_t prsm_tensor_make_view_range(const prsm_tensor_t *const t, const size_t *const shapeFrom, const size_t *const shapeTo);

/* 
    Tensor get/set value operations
*/

/**
 * @brief  Get value
 * @param  t tensor
 * @param  idx index
 * @returns prsm_float
 */
extern prsm_float prsm_tensor_get_val(const prsm_tensor_t *const t, const size_t idx);

/**
 * @brief  Set value
 * @param  t tensor
 * @param  idx index
 * @param  value value
 * @returns None
 */
extern void prsm_tensor_set_val(prsm_tensor_t *const t, const size_t idx, const prsm_float value);

/**
 * @brief  Set entire tensor to value
 * @param  t tensor
 * @param  value value
 * @returns None
 */
extern void prsm_tensor_set_all(prsm_tensor_t *const t, const prsm_float value);

/**
 * @brief  Set entire tensor to 1
 * @param  t tensor
 * @returns None
 */
extern void prsm_tensor_set_ones(prsm_tensor_t *const t);

/**
 * @brief  Set entire tensor to 0
 * @param  t tensor
 * @returns None
 */
extern void prsm_tensor_set_zeros(prsm_tensor_t *const t);

/**
 * @brief  Make identity
 * @param  t tensor
 * @returns None
 */
extern void prsm_tensor_set_identity(prsm_tensor_t *const t);

/* 
    Tensor-wise operations
*/

extern prsm_tensor_t *prsm_tensor_add(const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);
extern prsm_tensor_t *prsm_tensor_sub(const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);
extern prsm_tensor_t *prsm_tensor_mul(const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);
extern prsm_tensor_t *prsm_tensor_div(const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);

extern enum PrismaStatus prsm_tensor_add_into(prsm_tensor_t *const tout, const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);
extern enum PrismaStatus prsm_tensor_sub_into(prsm_tensor_t *const tout, const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);
extern enum PrismaStatus prsm_tensor_mul_into(prsm_tensor_t *const tout, const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);
extern enum PrismaStatus prsm_tensor_div_into(prsm_tensor_t *const tout, const prsm_tensor_t *const t1, const prsm_tensor_t *const t2);

/* 
    Tensor element-wise operations
*/

extern void prsm_tensor_apply_scale_add(prsm_tensor_t *const t, const prsm_float sval, const prsm_float aval);
extern void prsm_tensor_apply_ceil(prsm_tensor_t *const t);
extern void prsm_tensor_apply_floor(prsm_tensor_t *const t);
extern void prsm_tensor_apply_round(prsm_tensor_t *const t);
extern void prsm_tensor_apply_clip(prsm_tensor_t *const t, const prsm_float min, const prsm_float max);
extern void prsm_tensor_apply_abs(prsm_tensor_t *const t);
extern void prsm_tensor_apply_neg(prsm_tensor_t *const t);

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
    Tensor rand operations
*/

extern void prsm_tensor_rand(prsm_tensor_t *const t);
extern void prsm_tensor_rand_uniform(prsm_tensor_t *const t, const prsm_float lbound, const prsm_float ubound);
extern void prsm_tensor_rand_normal(prsm_tensor_t *const t, const prsm_float mu, const prsm_float sigma);
extern void prsm_tensor_rand_std_normal(prsm_tensor_t *const t);

#endif // PRISMA_CORE_TENSOR_H

