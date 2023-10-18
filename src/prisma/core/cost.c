#include "prisma/core/cost.h"

prsm_float prsm_cost_mae(const prsm_tensor_t *const input, const prsm_tensor_t *const target) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(input), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(target), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(prsm_tensor_match_shape(input, target), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_SHAPES));

    // calculate mae
    prsm_float sum = 0;
    const size_t size = prsm_tensor_size(input);
    VT_FOREACH(i, 0, size) {
        sum += PRSM_ABS(prsm_tensor_get_val(target, i) - prsm_tensor_get_val(input, i));
    }

    return sum/size;
}

prsm_float prsm_cost_mse(const prsm_tensor_t *const input, const prsm_tensor_t *const target) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(input), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(target), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(prsm_tensor_match_shape(input, target), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_SHAPES));

    // calculate mae
    prsm_float sum = 0;
    const size_t size = prsm_tensor_size(input);
    VT_FOREACH(i, 0, size) {
        sum += PRSM_POW(prsm_tensor_get_val(target, i) - prsm_tensor_get_val(input, i), 2);
    }

    return sum/size;
}

