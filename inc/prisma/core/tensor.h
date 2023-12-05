#ifndef PRISMA_CORE_TENSOR_H
#define PRISMA_CORE_TENSOR_H

/** TENSOR MODULE
 * This module is a collection of tensor and linear algebra functionality required by NN.

 * Functions:
    - prsm_tensor_create
    - prsm_tensor_create_ex
    - prsm_tensor_create_vec
    - prsm_tensor_create_mat
    - prsm_tensor_destroy
    - prsm_tensor_is_null
    - prsm_tensor_dim
    - prsm_tensor_shape
    - prsm_tensor_data
    - prsm_tensor_size
    - prsm_tensor_resize
    - prsm_tensor_resize_ex
    - prsm_tensor_dup
    - prsm_tensor_dup_into
    - prsm_tensor_transpose
    - prsm_tensor_flatten
    - prsm_tensor_diagflat
    - prsm_tensor_shapes_match
    - prsm_tensor_shapes_match_ex
    - prsm_tensor_equals
    - prsm_tensor_equals_approx
    - prsm_tensor_equals_array
    - prsm_tensor_assign
    - prsm_tensor_assign_array
    - prsm_tensor_swap
    - prsm_tensor_is_view
    - prsm_tensor_make_view
    - prsm_tensor_make_view_mat
    - prsm_tensor_make_view_vec
    - prsm_tensor_make_view_range
    - prsm_tensor_get_val
    - prsm_tensor_set_val
    - prsm_tensor_set_all
    - prsm_tensor_set_diag
    - prsm_tensor_set_ones
    - prsm_tensor_set_zeros
    - prsm_tensor_set_identity
    - prsm_tensor_set_from_array
    - prsm_tensor_sum
    - prsm_tensor_dot
    - prsm_tensor_vdot
    - prsm_tensor_add
    - prsm_tensor_sub
    - prsm_tensor_mul
    - prsm_tensor_apply_scale_add
    - prsm_tensor_apply_ceil
    - prsm_tensor_apply_floor
    - prsm_tensor_apply_round
    - prsm_tensor_apply_clip
    - prsm_tensor_apply_abs
    - prsm_tensor_apply_neg
    - prsm_tensor_apply_func
    - prsm_tensor_get_min
    - prsm_tensor_get_max
    - prsm_tensor_get_minmax
    - prsm_tensor_get_min_index
    - prsm_tensor_get_max_index
    - prsm_tensor_get_minmax_index
    - prsm_tensor_calc_sum
    - prsm_tensor_calc_prod
    - prsm_tensor_calc_mean
    - prsm_tensor_calc_var
    - prsm_tensor_calc_std
    - prsm_tensor_rand
    - prsm_tensor_rand_uniform
    - prsm_tensor_rand_normal
    - prsm_tensor_display
*/

#include "prisma/core/core.h"
#include "vita/math/math.h"
#include "vita/container/common.h"
#include "vita/allocator/mallocator.h"

typedef struct PrismaTensor {
    bool is_view;       // defines if tensor is modifiable or only viewable

    size_t ndim;        // number or dimensions: 1d, 2d, 3d, nd.
    size_t _shape;      // number of elements
    size_t *shape;      // tensor shape
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
extern prsm_tensor_t *prsm_tensor_create_ex(struct VitaBaseAllocatorType *const alloctr, const size_t ndim, const size_t shape[]);

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
 * @returns prsm_float *data
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
extern void prsm_tensor_resize_ex(prsm_tensor_t *const t, const size_t ndim, const size_t shape[]);

/**
 * @brief  Duplicates tensor
 * @param  t tensor
 * @returns tensor copy
 */
extern prsm_tensor_t *prsm_tensor_dup(const prsm_tensor_t *const t);

/**
 * @brief  Duplicates tensor into the preallocated tensor instance
 * @param  out copy to tensor (output)
 * @param  in copy from tensor (input)
 * @returns None
 */
extern void prsm_tensor_dup_into(prsm_tensor_t *const out, const prsm_tensor_t *const in);

/**
 * @brief  Transpose a tensor
 * @param  t tensor
 * @returns None
 */
extern void prsm_tensor_transpose(prsm_tensor_t *const t);

/**
 * @brief  Flattens a tensor into a vector
 * @param  t tensor
 * @returns None
 */
extern void prsm_tensor_flatten(prsm_tensor_t *const t);

/**
 * @brief  Diagonally flattens tensor elements
 * @param  t tensor
 * @returns None
 */
extern void prsm_tensor_diagflat(prsm_tensor_t *const t);

/* 
    Tensor data operations
*/

/**
 * @brief  Checks if shapes and dimensions match
 * @param  lhs tensor
 * @param  rhs tensor
 * @returns true if `lhs.ndim==rhs.ndim` and `lhs.shape==rhs.shape`
 */
extern bool prsm_tensor_shapes_match(const prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs);

/**
 * @brief  Checks if shapes and dimensions match from specified arguments
 * @param  t tensor
 * @param  ndim dimension
 * @param  shape shape
 * @returns true if `t.ndim==ndim` and `t.shape==shape`
 */
extern bool prsm_tensor_shapes_match_ex(const prsm_tensor_t *const t, const size_t ndim, const size_t shape[]);

/**
 * @brief  Checks if tensors are equal
 * @param  lhs tensor
 * @param  rhs tensor
 * @returns true if `lhs==rhs`
 */
extern bool prsm_tensor_equals(const prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs);

/**
 * @brief  Checks if tensors are approximately equal
 * @param  lhs tensor
 * @param  rhs tensor
 * @param  rtol relative tolerance value (-1 for automatic selection)
 * @returns true if `lhs==rhs`
 * 
 * @note if `rtol<0`, then `__FLT_EPSILON__` is used
 */
extern bool prsm_tensor_equals_approx(const prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs, const prsm_float rtol);

/**
 * @brief  Checks if tensor data equals to array values
 * @param  t tensor
 * @param  arr array
 * @param  arr_size array length
 * @returns true if `t.data == arr`
 */
extern bool prsm_tensor_equals_array(prsm_tensor_t *t, const prsm_float arr[], const size_t arr_size);

/**
 * @brief  Assigns values to `lhs` from `rhs`
 * @param  lhs tensor
 * @param  rhs tensor
 * @returns None
 */
extern void prsm_tensor_assign(prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs);

/**
 * @brief  Assign values from array
 * @param  t tensor
 * @param  arr array
 * @param  arr_size array length
 * @returns None
 */
extern void prsm_tensor_assign_array(prsm_tensor_t *t, const prsm_float arr[], const size_t arr_size);

/**
 * @brief  Swaps tensors
 * @param  lhs tensor
 * @param  rhs tensor
 * @returns None
 */
extern void prsm_tensor_swap(prsm_tensor_t *const lhs, prsm_tensor_t *const rhs);

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
 * @brief  Makes a view matrix given ndim matrix tensor
 * @param  t tensor
 * @param  dim view dimension
 * @returns prsm_tensor_t
 * 
 * @note it's a value type, no need to free it
 */
extern prsm_tensor_t prsm_tensor_make_view_mat(const prsm_tensor_t *const t, const size_t dim);

/**
 * @brief  Makes a view vector from 2d matrix row
 * @param  t tensor
 * @param  row matrix row index
 * @returns prsm_tensor_t
 * 
 * @note it's a value type, no need to free it
 */
extern prsm_tensor_t prsm_tensor_make_view_vec(const prsm_tensor_t *const t, const size_t row);

/**
 * @brief  Makes a range view from tensor
 * @param  t tensor
 * @param  range shape range { from, to }
 * @returns prsm_tensor_t
 * 
 * @note it's a value type, no need to free it
 */
extern prsm_tensor_t prsm_tensor_make_view_range(const prsm_tensor_t *const t, const size_t range[]);

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
 * @brief  Set diagonal elements to value
 * @param  t tensor
 * @param  value value
 * @returns None
 */
extern void prsm_tensor_set_diag(prsm_tensor_t *const t, const prsm_float value);

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

/**
 * @brief  Axis-wise summation
 * @param  out output tensor
 * @param  in tensor
 * @param  axis 2D { 0: row-wise, 1: column-wise }, 3D { 0: z-axis, 1: row-wise, 2: column-wise }
 * @returns prsm_tensor_t*
 * 
 * @note if `out==NULL`, tensor is allocated
 * @note `out` is zero initialized
*/
extern prsm_tensor_t *prsm_tensor_sum(prsm_tensor_t *out, const prsm_tensor_t *const in, const uint8_t axis);

/**
 * @brief  Add tensors
 * @param  out output tensor
 * @param  lhs tensor
 * @param  rhs tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==NULL`, tensor is allocated
 */
extern prsm_tensor_t *prsm_tensor_add(prsm_tensor_t *out, const prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs);

/**
 * @brief  Substract tensors
 * @param  out output tensor
 * @param  lhs tensor
 * @param  rhs tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==NULL`, tensor is allocated
 */
extern prsm_tensor_t *prsm_tensor_sub(prsm_tensor_t *out, const prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs);

/**
 * @brief  Tensor multiplication
 * @param  out output tensor
 * @param  lhs tensor
 * @param  rhs tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==NULL`, tensor is allocated
 * @note `out` is zero initialized
 */
extern prsm_tensor_t *prsm_tensor_dot(prsm_tensor_t *out, const prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs);

/**
 * @brief  Vector dot product
 * @param  lhs tensor
 * @param  rhs tensor
 * @returns prsm_float
 */
extern prsm_float prsm_tensor_vdot(const prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs);

/**
 * @brief  Element-wise multiplication (hadamard product)
 * @param  out output tensor
 * @param  lhs tensor
 * @param  rhs tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==NULL`, tensor is allocated
 */
extern prsm_tensor_t *prsm_tensor_mul(prsm_tensor_t *out, const prsm_tensor_t *const lhs, const prsm_tensor_t *const rhs);

/* 
    Tensor element-wise operations
*/

/**
 * @brief  Scale tensor element and add value
 * @param  t tensor
 * @param  sval scale value
 * @param  aval add value
 * @returns None
 */
extern void prsm_tensor_apply_scale_add(prsm_tensor_t *const t, const prsm_float sval, const prsm_float aval);

/**
 * @brief  Apply ceiling operation 
 * @param  t tensor
 * @returns None
 */
extern void prsm_tensor_apply_ceil(prsm_tensor_t *const t);

/**
 * @brief  Apply floor operation 
 * @param  t tensor
 * @returns None
 */
extern void prsm_tensor_apply_floor(prsm_tensor_t *const t);

/**
 * @brief  Round off tensor values 
 * @param  t tensor
 * @returns None
 */
extern void prsm_tensor_apply_round(prsm_tensor_t *const t);

/**
 * @brief  Clip tensor values 
 * @param  t tensor
 * @param  min lower bound
 * @param  max upper bound
 * @returns None
 */
extern void prsm_tensor_apply_clip(prsm_tensor_t *const t, const prsm_float min, const prsm_float max);

/**
 * @brief  Apply abs function 
 * @param  t tensor
 * @returns None
 */
extern void prsm_tensor_apply_abs(prsm_tensor_t *const t);

/**
 * @brief  Negate tensor values 
 * @param  t tensor
 * @returns None
 */
extern void prsm_tensor_apply_neg(prsm_tensor_t *const t);

/**
 * @brief  Apply function to tensor values 
 * @param  t tensor
 * @param  func custom function
 * @returns None
 */
extern void prsm_tensor_apply_func(prsm_tensor_t *const t, prsm_float (*func)(prsm_float));

/* 
    Tensor statistics on the whole tensor
*/

/**
 * @brief  Find minimum value 
 * @param  t tensor
 * @returns min
 */
extern prsm_float prsm_tensor_get_min(const prsm_tensor_t *const t);

/**
 * @brief  Find maximum value 
 * @param  t tensor
 * @returns max
 */
extern prsm_float prsm_tensor_get_max(const prsm_tensor_t *const t);

/**
 * @brief  Find minimum and maximum value 
 * @param  t tensor
 * @param  min save min value
 * @param  min save max value
 * @returns None
 */
extern void prsm_tensor_get_minmax(const prsm_tensor_t *const t, prsm_float *min, prsm_float *max);

/**
 * @brief  Find index of minimum value 
 * @param  t tensor
 * @returns index
 */
extern size_t prsm_tensor_get_min_index(const prsm_tensor_t *const t);

/**
 * @brief  Find index of maximum value 
 * @param  t tensor
 * @returns index
 */
extern size_t prsm_tensor_get_max_index(const prsm_tensor_t *const t);

/**
 * @brief  Find index of minimum and maximum value 
 * @param  t tensor
 * @param  min_index save min index value
 * @param  max_index save max index value
 * @returns None
 */
extern void prsm_tensor_get_minmax_index(const prsm_tensor_t *const t, size_t *min_index, size_t *max_index);

/**
 * @brief  Calculate sum 
 * @param  t tensor
 * @returns sum
 */
extern prsm_float prsm_tensor_calc_sum(const prsm_tensor_t *const t);

/**
 * @brief  Calculate product 
 * @param  t tensor
 * @returns product
 */
extern prsm_float prsm_tensor_calc_prod(const prsm_tensor_t *const t);

/**
 * @brief  Calculate mean 
 * @param  t tensor
 * @returns mean
 */
extern prsm_float prsm_tensor_calc_mean(const prsm_tensor_t *const t);

/**
 * @brief  Calculate variance 
 * @param  t tensor
 * @returns variance
 */
extern prsm_float prsm_tensor_calc_var(const prsm_tensor_t *const t);

/**
 * @brief  Calculate standard deviation 
 * @param  t tensor
 * @returns standard deviation
 */
extern prsm_float prsm_tensor_calc_std(const prsm_tensor_t *const t);

/* 
    Tensor rand operations
*/

/**
 * @brief  Randomize tensor values between [0; 1]
 * @param  t tensor
 * @returns None
 */
extern void prsm_tensor_rand(prsm_tensor_t *const t);

/**
 * @brief  Randomize tensor values uniformely between [lbound; ubound)
 * @param  t tensor
 * @param  lbound lower bound
 * @param  ubound upper bound
 * @returns None
 */
extern void prsm_tensor_rand_uniform(prsm_tensor_t *const t, const prsm_float lbound, const prsm_float ubound);

/**
 * @brief  Randomize tensor values from normal distribution
 * @param  t tensor
 * @param  mu mean
 * @param  std sigma
 * @returns None
 */
extern void prsm_tensor_rand_normal(prsm_tensor_t *const t, const prsm_float mu, const prsm_float std);

/* 
    Pretty printing
*/

/**
 * @brief  Pretty printing 
 * @param  t tensor
 * @param  range view
 * @returns None
 */
extern void prsm_tensor_display(const prsm_tensor_t *const t, const size_t range[]);

#endif // PRISMA_CORE_TENSOR_H

