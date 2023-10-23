#ifndef PRISMA_CORE_MATH_H
#define PRISMA_CORE_MATH_H

/** MATH MODULE
 * This module is a collection of common math functions and its derivates.

 * Functions:
    - prsm_math_sigmoid
    - prsm_math_sigmoid_d
    - prsm_math_tanh
    - prsm_math_tanh_d
    - prsm_math_linear
    - prsm_math_linear_d
    - prsm_math_ramp
    - prsm_math_ramp_d
    - prsm_math_htanh
    - prsm_math_htanh_d
    - prsm_math_relu
    - prsm_math_relu_d
    - prsm_math_lrelu
    - prsm_math_lrelu_d
    - prsm_math_elu
    - prsm_math_elu_d
    - prsm_math_selu
    - prsm_math_selu_d
    - prsm_math_prelu
    - prsm_math_prelu_d    
*/

#include "prisma/core/core.h"

/**
 * @brief  Sigmoid function
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_math_sigmoid(const prsm_float x);

/**
 * @brief  Derivative of sigmoid function
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_math_sigmoid_d(const prsm_float x);

/**
 * @brief  Tanh function
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_math_tanh(const prsm_float x);

/**
 * @brief  Derivative of tanh function
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_math_tanh_d(const prsm_float x);

/**
 * @brief  Linear function
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_math_linear(const prsm_float x);

/**
 * @brief  Derivative of linear function
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_math_linear_d(const prsm_float x);

/**
 * @brief  Ramp function
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_math_ramp(prsm_float x);

/**
 * @brief  Derivative of ramp function
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_math_ramp_d(prsm_float x);

/**
 * @brief  Hard tanh function
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_math_htanh(prsm_float x);

/**
 * @brief  Derivative of hard tanh function
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_math_htanh_d(prsm_float x);

/**
 * @brief  RELU function
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_math_relu(const prsm_float x);

/**
 * @brief  Derivative of RELU function
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_math_relu_d(const prsm_float x);

/**
 * @brief  Leaky RELU function
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_math_lrelu(const prsm_float x);

/**
 * @brief  Derivative of Leaky RELU function
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_math_lrelu_d(const prsm_float x);

/**
 * @brief  ELU function
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_math_elu(const prsm_float x);

/**
 * @brief  Derivative of ELU function
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_math_elu_d(const prsm_float x);

/**
 * @brief  SELU function
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_math_selu(const prsm_float x);

/**
 * @brief  Derivative of SELU function
 * @param  x input value
 * @returns prsm_float
 */
extern prsm_float prsm_math_selu_d(const prsm_float x);

/**
 * @brief  Parametric RELU function
 * @param  x input value
 * @param  a alpha learning rate
 * @returns prsm_float
 */
extern prsm_float prsm_math_prelu(const prsm_float x, const prsm_float a);

/**
 * @brief  Derivative of Parametric RELU function
 * @param  x input value
 * @param  a alpha learning rate
 * @returns prsm_float
 */
extern prsm_float prsm_math_prelu_d(const prsm_float x, const prsm_float a);

#endif // PRISMA_CORE_MATH_H

