#ifndef PRISMA_CORE_COST_H
#define PRISMA_CORE_COST_H

/** COST MODULE
 * This module contains popular cost functions.

 * Functions:
    - prsm_cost_mae
    - prsm_cost_mse
*/

#include "prisma/core/core.h"
#include "prisma/core/tensor.h"

/**
 * @brief  Mean absolute error
 * @param  input predicted values
 * @param  target actual values
 * @returns prsm_float
 */
extern prsm_float prsm_cost_mae(const prsm_tensor_t *const input, const prsm_tensor_t *const target);

/**
 * @brief  Mean squared error
 * @param  input predicted values
 * @param  target actual values
 * @returns prsm_float
 */
extern prsm_float prsm_cost_mse(const prsm_tensor_t *const input, const prsm_tensor_t *const target);

#endif // PRISMA_CORE_COST_H

