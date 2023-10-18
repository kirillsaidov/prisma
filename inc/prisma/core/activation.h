#ifndef PRISMA_CORE_ACTIVATION_H
#define PRISMA_CORE_ACTIVATION_H

/** CORE MODULE
 * This module is a collection of common activation functions and its derivates.

 * Functions:
    - prsm_activation_sigmoid
    - prsm_activation_sigmoid_d
    - prsm_activation_tanh
    - prsm_activation_tanh_d
    - prsm_activation_linear
    - prsm_activation_linear_d
    - prsm_activation_ramp
    - prsm_activation_ramp_d
    - prsm_activation_relu
    - prsm_activation_relu_d
    - prsm_activation_lrelu
    - prsm_activation_lrelu_d
    - prsm_activation_elu
    - prsm_activation_elu_d
    - prsm_activation_selu
    - prsm_activation_selu_d
    - prsm_activation_prelu
    - prsm_activation_prelu_d    
*/

#include "prisma/core/core.h"

/**
 * @brief  Sigmoid activation
 * @param  v input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_sigmoid(const prsm_float v);

/**
 * @brief  Derivative of sigmoid activation
 * @param  v input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_sigmoid_d(const prsm_float v);

/**
 * @brief  Tanh activation
 * @param  v input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_tanh(const prsm_float v);

/**
 * @brief  Derivative of tanh activation
 * @param  v input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_tahn_d(const prsm_float v);

/**
 * @brief  Linear activation
 * @param  v input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_linear(const prsm_float v);

/**
 * @brief  Ramp activation
 * @param  v input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_ramp(prsm_float v);

/**
 * @brief  Derivative of ramp activation
 * @param  v input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_ramp_d(prsm_float v);

/**
 * @brief  Derivative of linear activation
 * @param  v input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_linear_d(const prsm_float v);

/**
 * @brief  RELU activation
 * @param  v input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_relu(const prsm_float v);

/**
 * @brief  Derivative of RELU activation
 * @param  v input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_relu_d(const prsm_float v);

/**
 * @brief  Leaky RELU activation
 * @param  v input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_lrelu(const prsm_float v);

/**
 * @brief  Derivative of Leaky RELU activation
 * @param  v input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_lrelu_d(const prsm_float v);

/**
 * @brief  ELU activation
 * @param  v input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_elu(const prsm_float v);

/**
 * @brief  Derivative of ELU activation
 * @param  v input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_elu_d(const prsm_float v);

/**
 * @brief  SELU activation
 * @param  v input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_selu(const prsm_float v);

/**
 * @brief  Derivative of SELU activation
 * @param  v input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_selu_d(const prsm_float v);

/**
 * @brief  Parametric RELU activation
 * @param  v input value
 * @param  a alpha learning rate
 * @returns prsm_float
 */
extern prsm_float prsm_activation_prelu(const prsm_float v, const prsm_float a);

/**
 * @brief  Derivative of Parametric RELU activation
 * @param  v input value
 * @param  a alpha learning rate
 * @returns prsm_float
 */
extern prsm_float prsm_activation_prelu_d(const prsm_float v, const prsm_float a);

#endif // PRISMA_CORE_ACTIVATION_H

