#include "prisma/core/loss.h"

prsm_float prsm_loss_mae(const prsm_tensor_t *const input, const prsm_tensor_t *const target) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(input), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(target), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(prsm_tensor_match_shape(input, target), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_SHAPES));

    // calculate mae
    prsm_float sum = 0;
    const prsm_float *y = prsm_tensor_data(target);
    const prsm_float *yhat = prsm_tensor_data(input);
    const size_t size = prsm_tensor_size(input);
    VT_FOREACH(i, 0, size) {
        sum += PRSM_ABS(y[i] - yhat[i]);
    }

    return sum/size;
}

prsm_float prsm_loss_mse(const prsm_tensor_t *const input, const prsm_tensor_t *const target) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(input), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(target), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(prsm_tensor_match_shape(input, target), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_SHAPES));

    // calculate mae
    prsm_float sum = 0;
    const prsm_float *y = prsm_tensor_data(target);
    const prsm_float *yhat = prsm_tensor_data(input);
    const size_t size = prsm_tensor_size(input);
    VT_FOREACH(i, 0, size) {
        sum += PRSM_POW(y[i] - yhat[i], 2);
    }

    return sum/size;
}

prsm_float prsm_loss_rmse(const prsm_tensor_t *const input, const prsm_tensor_t *const target) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(input), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(target), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(prsm_tensor_match_shape(input, target), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_SHAPES));

    return PRSM_SQRT(prsm_loss_mse(input, target));
}

prsm_float prsm_loss_bce(const prsm_tensor_t *const input, const prsm_tensor_t *const target) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(input), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(target), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(prsm_tensor_match_shape(input, target), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_SHAPES));

    // calculate bce
    prsm_float sum = 0;
    const prsm_float *y = prsm_tensor_data(target);
    const prsm_float *yhat = prsm_tensor_data(input);
    const size_t size = prsm_tensor_size(input);
    VT_FOREACH(i, 0, size) {
        const prsm_float yhat_i = PRSM_CLAMP(yhat[i], 1e-7, 1 - 1e-7);
        sum += y[i] * PRSM_LOG(yhat_i + 1e-7) + (1 - y[i]) * PRSM_LOG(1 - yhat_i + 1e-7);
    }

    return -sum/size;
}

prsm_float prsm_loss_cce(const prsm_tensor_t *const input, const prsm_tensor_t *const target) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(input), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(target), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(prsm_tensor_match_shape(input, target), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INCOMPATIBLE_SHAPES));

    // calculate bce
    prsm_float sum = 0;
    const prsm_float *y = prsm_tensor_data(target);
    const prsm_float *yhat = prsm_tensor_data(input);
    const size_t size = prsm_tensor_size(input);
    VT_FOREACH(i, 0, size) {
        sum += y[i] * PRSM_LOG(yhat[i]);
    }

    return -sum;
}

