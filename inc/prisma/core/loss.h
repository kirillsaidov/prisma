#ifndef PRISMA_CORE_LOSS_H
#define PRISMA_CORE_LOSS_H

/** COST MODULE
 * This module contains popular cost functions.

 * Functions:
    - prsm_loss_mae
    - prsm_loss_mae_d
    - prsm_loss_mse
    - prsm_loss_mse_d
    - prsm_loss_rmse
    - prsm_loss_rmse_d
    - prsm_loss_bce
    - prsm_loss_bce_d
    - prsm_loss_cce
*/

#include "prisma/core/core.h"
#include "prisma/core/tensor.h"

/**
 * @brief  Mean absolute error
 * @param  input predicted values
 * @param  target actual values
 * @returns prsm_float
 */
extern prsm_float prsm_loss_mae(const prsm_tensor_t *const input, const prsm_tensor_t *const target);

/**
 * @brief  Derivative of mean absolute error
 * @param  out output tensor
 * @param  input predicted values
 * @param  target actual values
 * @returns prsm_tensor_t*
 */
extern prsm_tensor_t *prsm_loss_mae_d(prsm_tensor_t *out, const prsm_tensor_t *const input, const prsm_tensor_t *const target);

/**
 * @brief  Mean squared error
 * @param  input predicted values
 * @param  target actual values
 * @returns prsm_float
 */
extern prsm_float prsm_loss_mse(const prsm_tensor_t *const input, const prsm_tensor_t *const target);

/**
 * @brief  Derivative of mean squared error
 * @param  out output tensor
 * @param  input predicted values
 * @param  target actual values
 * @returns prsm_tensor_t*
 */
extern prsm_tensor_t *prsm_loss_mse_d(prsm_tensor_t *out, const prsm_tensor_t *const input, const prsm_tensor_t *const target);

/**
 * @brief  Root mean squared error
 * @param  input predicted values
 * @param  target actual values
 * @returns prsm_float
 */
extern prsm_float prsm_loss_rmse(const prsm_tensor_t *const input, const prsm_tensor_t *const target);

/**
 * @brief  Derivative of root mean squared error
 * @param  out output tensor
 * @param  input predicted values
 * @param  target actual values
 * @returns prsm_tensor_t*
 */
extern prsm_tensor_t *prsm_loss_rmse_d(prsm_tensor_t *out, const prsm_tensor_t *const input, const prsm_tensor_t *const target);

/**
 * @brief  Binary cross entropy loss
 * @param  input predicted values
 * @param  target actual values
 * @returns prsm_float
 */
extern prsm_float prsm_loss_bce(const prsm_tensor_t *const input, const prsm_tensor_t *const target);

/**
 * @brief  Derivative of binary cross entropy loss
 * @param  out output tensor
 * @param  input predicted values
 * @param  target actual values
 * @returns prsm_tensor_t*
 */
extern prsm_tensor_t *prsm_loss_bce_d(prsm_tensor_t *out, const prsm_tensor_t *const input, const prsm_tensor_t *const target);

/**
 * @brief  Categorical cross entropy loss
 * @param  input predicted values
 * @param  target actual values
 * @returns prsm_float
 */
extern prsm_float prsm_loss_cce(const prsm_tensor_t *const input, const prsm_tensor_t *const target);

/**
 * @brief  Derivative of categorical cross entropy loss
 * @param  out output tensor
 * @param  input predicted values
 * @param  target actual values
 * @returns prsm_tensor_t*
 */
extern prsm_tensor_t *prsm_loss_cce_d(prsm_tensor_t *out, const prsm_tensor_t *const input, const prsm_tensor_t *const target);

#endif // PRISMA_CORE_LOSS_H

