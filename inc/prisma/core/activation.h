#ifndef PRISMA_CORE_ACTIVATION_H
#define PRISMA_CORE_ACTIVATION_H

/** ACTIVATION MODULE
 * This module is a collection of common activation functions and its derivates.

 * Functions:
    - prsm_activate_sigmoid
    // - prsm_activate_sigmoid_d
    - prsm_activate_tanh
    // - prsm_activate_tanh_d
    - prsm_activate_linear
    // - prsm_activate_linear_d
    - prsm_activate_ramp
    // - prsm_activate_ramp_d
    - prsm_activate_htanh
    // - prsm_activate_htanh_d
    - prsm_activate_relu
    // - prsm_activate_relu_d
    // - prsm_activate_lrelu
    // - prsm_activate_lrelu_d
    - prsm_activate_elu
    // - prsm_activate_elu_d
    - prsm_activate_selu
    // - prsm_activate_selu_d
    - prsm_activate_prelu
    // - prsm_activate_prelu_d  
    - prsm_activate_softmax
    // - prsm_activate_softmax_d   
    - prsm_activate_ssoftmax
    // - prsm_activate_softmax_d
    - prsm_activate_lsoftmax
    // - prsm_activate_softmax_d
*/

#include "prisma/core/core.h"
#include "prisma/core/math.h"
#include "prisma/core/tensor.h"

/**
 * @brief  Sigmoid activation
 * @param  out output tensor
 * @param  in input tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==NULL`, tensor is allocated
 */
extern prsm_tensor_t *prsm_activate_sigmoid(prsm_tensor_t *out, const prsm_tensor_t *const in);

/**
 * @brief  Derivative of sigmoid activation
 * @param  out output tensor
 * @param  in input tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==null`, tensor is allocated
 */
// extern prsm_tensor_t *prsm_activate_sigmoid_d(prsm_tensor_t *out, const prsm_tensor_t *const in);

/**
 * @brief  Tanh activation
 * @param  out output tensor
 * @param  in input tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==null`, tensor is allocated
 */
extern prsm_tensor_t *prsm_activate_tanh(prsm_tensor_t *out, const prsm_tensor_t *const in);

/**
 * @brief  Derivative of tanh activation
 * @param  out output tensor
 * @param  in input tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==null`, tensor is allocated
 */
// extern prsm_tensor_t *prsm_activate_tahn_d(prsm_tensor_t *out, const prsm_tensor_t *const in);

/**
 * @brief  Linear activation
 * @param  out output tensor
 * @param  in input tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==null`, tensor is allocated
 */
extern prsm_tensor_t *prsm_activate_linear(prsm_tensor_t *out, const prsm_tensor_t *const in);

/**
 * @brief  Derivative of linear activation
 * @param  out output tensor
 * @param  in input tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==null`, tensor is allocated
 */
// extern prsm_tensor_t *prsm_activate_linear_d(prsm_tensor_t *out, const prsm_tensor_t *const in);

/**
 * @brief  Ramp activation
 * @param  out output tensor
 * @param  in input tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==null`, tensor is allocated
 */
extern prsm_tensor_t *prsm_activate_ramp(prsm_tensor_t *out, const prsm_tensor_t *const in);

/**
 * @brief  Derivative of ramp activation
 * @param  out output tensor
 * @param  in input tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==null`, tensor is allocated
 */
// extern prsm_tensor_t *prsm_activate_ramp_d(prsm_tensor_t *out, const prsm_tensor_t *const in);

/**
 * @brief  Hard tanh activation
 * @param  out output tensor
 * @param  in input tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==null`, tensor is allocated
 */
extern prsm_tensor_t *prsm_activate_htanh(prsm_tensor_t *out, const prsm_tensor_t *const in);

/**
 * @brief  Derivative of hard tanh activation
 * @param  out output tensor
 * @param  in input tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==null`, tensor is allocated
 */
// extern prsm_tensor_t *prsm_activate_htanh_d(prsm_tensor_t *out, const prsm_tensor_t *const in);

/**
 * @brief  RELU activation
 * @param  out output tensor
 * @param  in input tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==null`, tensor is allocated
 */
extern prsm_tensor_t *prsm_activate_relu(prsm_tensor_t *out, const prsm_tensor_t *const in);

/**
 * @brief  Derivative of RELU activation
 * @param  out output tensor
 * @param  in input tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==null`, tensor is allocated
 */
// extern prsm_tensor_t *prsm_activate_relu_d(prsm_tensor_t *out, const prsm_tensor_t *const in);

/**
 * @brief  Leaky RELU activation
 * @param  out output tensor
 * @param  in input tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==null`, tensor is allocated
 */
extern prsm_tensor_t *prsm_activate_lrelu(prsm_tensor_t *out, const prsm_tensor_t *const in);

/**
 * @brief  Derivative of Leaky RELU activation
 * @param  out output tensor
 * @param  in input tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==null`, tensor is allocated
 */
// extern prsm_tensor_t *prsm_activate_lrelu_d(prsm_tensor_t *out, const prsm_tensor_t *const in);

/**
 * @brief  ELU activation
 * @param  out output tensor
 * @param  in input tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==null`, tensor is allocated
 */
extern prsm_tensor_t *prsm_activate_elu(prsm_tensor_t *out, const prsm_tensor_t *const in);

/**
 * @brief  Derivative of ELU activation
 * @param  out output tensor
 * @param  in input tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==null`, tensor is allocated
 */
// extern prsm_tensor_t *prsm_activate_elu_d(prsm_tensor_t *out, const prsm_tensor_t *const in);

/**
 * @brief  SELU activation
 * @param  out output tensor
 * @param  in input tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==null`, tensor is allocated
 */
extern prsm_tensor_t *prsm_activate_selu(prsm_tensor_t *out, const prsm_tensor_t *const in);

/**
 * @brief  Derivative of SELU activation
 * @param  out output tensor
 * @param  in input tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==null`, tensor is allocated
 */
// extern prsm_tensor_t *prsm_activate_selu_d(prsm_tensor_t *out, const prsm_tensor_t *const in);

/**
 * @brief  Parametric RELU activation
 * @param  x input value
 * @param  a alpha learning rate
 * @returns prsm_float
 */
extern prsm_tensor_t *prsm_activate_prelu(prsm_tensor_t *out, const prsm_tensor_t *const in, const prsm_float a);

/**
 * @brief  Derivative of Parametric RELU activation
 * @param  x input value
 * @param  a alpha learning rate
 * @returns prsm_float
 */
// extern prsm_tensor_t *prsm_activate_prelu_d(prsm_tensor_t *out, const prsm_tensor_t *const in, const prsm_float a);

/**
 * @brief  Softmax: Normalizes input to [0; 1] for multi-class classification
 * @param  out output tensor
 * @param  in input tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==NULL`, tensor is allocated
 */
extern prsm_tensor_t *prsm_activate_softmax(prsm_tensor_t *out, const prsm_tensor_t *const in);

/**
 * @brief  Stable softmax: shift-normalizes input to [0; 1] for multi-class classification for numerical stability
 * @param  out output tensor
 * @param  in input tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==NULL`, tensor is allocated
 */
extern prsm_tensor_t *prsm_activate_ssoftmax(prsm_tensor_t *out, const prsm_tensor_t *const in);

/**
 * @brief  Log softmax
 * @param  out output tensor
 * @param  in input tensor
 * @returns prsm_tensor_t*
 * 
 * @note if `out==NULL`, tensor is allocated
 */
extern prsm_tensor_t *prsm_activate_lsoftmax(prsm_tensor_t *out, const prsm_tensor_t *const in);

#endif // PRISMA_CORE_ACTIVATION_H

