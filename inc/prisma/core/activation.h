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
    - prsm_activation_htanh
    - prsm_activation_htanh_d
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
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_sigmoid(const prsm_float x);

/**
 * @brief  Derivative of sigmoid activation
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_sigmoid_d(const prsm_float x);

/**
 * @brief  Tanh activation
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_tanh(const prsm_float x);

/**
 * @brief  Derivative of tanh activation
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_tahn_d(const prsm_float x);

/**
 * @brief  Linear activation
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_linear(const prsm_float x);

/**
 * @brief  Derivative of linear activation
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_linear_d(const prsm_float x);

/**
 * @brief  Ramp activation
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_ramp(prsm_float x);

/**
 * @brief  Derivative of ramp activation
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_ramp_d(prsm_float x);

/**
 * @brief  Hard tanh activation
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_htanh(prsm_float x);

/**
 * @brief  Derivative of hard tanh activation
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_htanh_d(prsm_float x);

/**
 * @brief  RELU activation
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_relu(const prsm_float x);

/**
 * @brief  Derivative of RELU activation
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_relu_d(const prsm_float x);

/**
 * @brief  Leaky RELU activation
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_lrelu(const prsm_float x);

/**
 * @brief  Derivative of Leaky RELU activation
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_lrelu_d(const prsm_float x);

/**
 * @brief  ELU activation
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_elu(const prsm_float x);

/**
 * @brief  Derivative of ELU activation
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_elu_d(const prsm_float x);

/**
 * @brief  SELU activation
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_selu(const prsm_float x);

/**
 * @brief  Derivative of SELU activation
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_activation_selu_d(const prsm_float x);

/**
 * @brief  Parametric RELU activation
 * @param  x input value
 * @param  a alpha learning rate
 * @returns prsm_float
 */
extern prsm_float prsm_activation_prelu(const prsm_float x, const prsm_float a);

/**
 * @brief  Derivative of Parametric RELU activation
 * @param  x input value
 * @param  a alpha learning rate
 * @returns prsm_float
 */
extern prsm_float prsm_activation_prelu_d(const prsm_float x, const prsm_float a);

#endif // PRISMA_CORE_ACTIVATION_H

