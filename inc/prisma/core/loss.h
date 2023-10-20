#ifndef PRISMA_CORE_LOSS_H
#define PRISMA_CORE_LOSS_H

/** COST MODULE
 * This module contains popular cost functions.

 * Functions:
    - prsm_loss_mae
    - prsm_loss_mse
    - prsm_loss_rmse
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
 * @brief  Mean squared error
 * @param  input predicted values
 * @param  target actual values
 * @returns prsm_float
 */
extern prsm_float prsm_loss_mse(const prsm_tensor_t *const input, const prsm_tensor_t *const target);

/**
 * @brief  Root mean squared error
 * @param  input predicted values
 * @param  target actual values
 * @returns prsm_float
 */
extern prsm_float prsm_loss_rmse(const prsm_tensor_t *const input, const prsm_tensor_t *const target);

/**
 * @brief  Binary cross entropy loss
 * @param  input predicted values
 * @param  target actual values
 * @returns prsm_float
 */
extern prsm_float prsm_loss_bce(const prsm_tensor_t *const input, const prsm_tensor_t *const target);

/**
 * @brief  Categorical cross entropy loss
 * @param  input predicted values
 * @param  target actual values
 * @returns prsm_float
 */
extern prsm_float prsm_loss_cce(const prsm_tensor_t *const input, const prsm_tensor_t *const target);

#endif // PRISMA_CORE_LOSS_H

