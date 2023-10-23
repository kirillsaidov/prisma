#include "prisma/core/activation.h"

prsm_tensor_t *prsm_activate_sigmoid(prsm_tensor_t *out, const prsm_tensor_t *const in) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(in), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // create tensor
    prsm_tensor_t *ret = (out == NULL)
        ? prsm_tensor_create_shape(in->alloctr, in->ndim, in->shape)
        : out;

    // check size
    if (!prsm_tensor_match_shape(ret, in)) {
        prsm_tensor_resize_shape(ret, in->ndim, in->shape);
    }

    // copy over the values 
    prsm_tensor_assign(ret, in);

    // activate
    prsm_tensor_apply_func(ret, prsm_math_sigmoid);

    return ret;
}

// prsm_tensor_t *prsm_activate_sigmoid_d(prsm_tensor_t *out, const prsm_tensor_t *const in) {
//     const prsm_float s = prsm_activate_sigmoid(x);
//     return s * (1 - s);
// }

prsm_tensor_t *prsm_activate_tanh(prsm_tensor_t *out, const prsm_tensor_t *const in) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(in), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // create tensor
    prsm_tensor_t *ret = (out == NULL)
        ? prsm_tensor_create_shape(in->alloctr, in->ndim, in->shape)
        : out;

    // check size
    if (!prsm_tensor_match_shape(ret, in)) {
        prsm_tensor_resize_shape(ret, in->ndim, in->shape);
    }

    // copy over the values 
    prsm_tensor_assign(ret, in);

    // activate
    prsm_tensor_apply_func(ret, prsm_math_tanh);

    return ret;
}

// prsm_tensor_t *prsm_activate_tahn_d(prsm_tensor_t *out, const prsm_tensor_t *const in) {
//     const prsm_float s = prsm_activate_tanh(x);
//     return 1 - (s * s);
// }

prsm_tensor_t *prsm_activate_linear(prsm_tensor_t *out, const prsm_tensor_t *const in) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(in), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // create tensor
    prsm_tensor_t *ret = (out == NULL)
        ? prsm_tensor_create_shape(in->alloctr, in->ndim, in->shape)
        : out;

    // check size
    if (!prsm_tensor_match_shape(ret, in)) {
        prsm_tensor_resize_shape(ret, in->ndim, in->shape);
    }

    // copy over the values 
    prsm_tensor_assign(ret, in);

    return ret;
}

// prsm_tensor_t *prsm_activate_linear_d(prsm_tensor_t *out, const prsm_tensor_t *const in) {
//     (void)x;
//     return 1;
// }

prsm_tensor_t *prsm_activate_ramp(prsm_tensor_t *out, const prsm_tensor_t *const in) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(in), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // create tensor
    prsm_tensor_t *ret = (out == NULL)
        ? prsm_tensor_create_shape(in->alloctr, in->ndim, in->shape)
        : out;

    // check size
    if (!prsm_tensor_match_shape(ret, in)) {
        prsm_tensor_resize_shape(ret, in->ndim, in->shape);
    }

    // copy over the values 
    prsm_tensor_assign(ret, in);

    // activate
    prsm_tensor_apply_func(ret, prsm_math_ramp);

    return ret;
}

// prsm_tensor_t *prsm_activate_ramp_d(prsm_tensor_t *out, const prsm_tensor_t *const in) {
//     return (x > 0) + 0.1;
// }

prsm_tensor_t *prsm_activate_htanh(prsm_tensor_t *out, const prsm_tensor_t *const in) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(in), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // create tensor
    prsm_tensor_t *ret = (out == NULL)
        ? prsm_tensor_create_shape(in->alloctr, in->ndim, in->shape)
        : out;

    // check size
    if (!prsm_tensor_match_shape(ret, in)) {
        prsm_tensor_resize_shape(ret, in->ndim, in->shape);
    }

    // copy over the values 
    prsm_tensor_assign(ret, in);

    // activate
    prsm_tensor_apply_func(ret, prsm_math_htanh);

    return ret;
}

// prsm_tensor_t *prsm_activate_htanh_d(prsm_tensor_t *out, const prsm_tensor_t *const in) {
//     if (x > -1 && x < 1) return 1;
//     return 0;
// }

prsm_tensor_t *prsm_activate_relu(prsm_tensor_t *out, const prsm_tensor_t *const in) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(in), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // create tensor
    prsm_tensor_t *ret = (out == NULL)
        ? prsm_tensor_create_shape(in->alloctr, in->ndim, in->shape)
        : out;

    // check size
    if (!prsm_tensor_match_shape(ret, in)) {
        prsm_tensor_resize_shape(ret, in->ndim, in->shape);
    }

    // copy over the values 
    prsm_tensor_assign(ret, in);

    // activate
    prsm_tensor_apply_func(ret, prsm_math_relu);

    return ret;
}

// prsm_tensor_t *prsm_activate_relu_d(prsm_tensor_t *out, const prsm_tensor_t *const in) {
//     return x >= 0 ? 1 : 0;
// }

prsm_tensor_t *prsm_activate_lrelu(prsm_tensor_t *out, const prsm_tensor_t *const in) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(in), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // create tensor
    prsm_tensor_t *ret = (out == NULL)
        ? prsm_tensor_create_shape(in->alloctr, in->ndim, in->shape)
        : out;

    // check size
    if (!prsm_tensor_match_shape(ret, in)) {
        prsm_tensor_resize_shape(ret, in->ndim, in->shape);
    }

    // copy over the values 
    prsm_tensor_assign(ret, in);

    // activate
    prsm_tensor_apply_func(ret, prsm_math_lrelu);

    return ret;
}

// prsm_tensor_t *prsm_activate_lrelu_d(prsm_tensor_t *out, const prsm_tensor_t *const in) {
//     return x >= 0 ? 1 : 0.01;
// }

prsm_tensor_t *prsm_activate_elu(prsm_tensor_t *out, const prsm_tensor_t *const in) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(in), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // create tensor
    prsm_tensor_t *ret = (out == NULL)
        ? prsm_tensor_create_shape(in->alloctr, in->ndim, in->shape)
        : out;

    // check size
    if (!prsm_tensor_match_shape(ret, in)) {
        prsm_tensor_resize_shape(ret, in->ndim, in->shape);
    }

    // copy over the values 
    prsm_tensor_assign(ret, in);

    // activate
    prsm_tensor_apply_func(ret, prsm_math_elu);

    return ret;
}

// prsm_tensor_t *prsm_activate_elu_d(prsm_tensor_t *out, const prsm_tensor_t *const in) {
//     return x > 0 ? 1 : PRSM_EXP(x);
// }

prsm_tensor_t *prsm_activate_selu(prsm_tensor_t *out, const prsm_tensor_t *const in) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(in), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // create tensor
    prsm_tensor_t *ret = (out == NULL)
        ? prsm_tensor_create_shape(in->alloctr, in->ndim, in->shape)
        : out;

    // check size
    if (!prsm_tensor_match_shape(ret, in)) {
        prsm_tensor_resize_shape(ret, in->ndim, in->shape);
    }

    // copy over the values 
    prsm_tensor_assign(ret, in);

    // activate
    prsm_tensor_apply_func(ret, prsm_math_selu);

    return ret;
}

// prsm_tensor_t *prsm_activate_selu_d(prsm_tensor_t *out, const prsm_tensor_t *const in) {
//     return x > 0 ? 1.0507 : (1.0507 * 1.6732 * PRSM_EXP(x));
// }

prsm_tensor_t *prsm_activate_prelu(prsm_tensor_t *out, const prsm_tensor_t *const in, const prsm_float a) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(in), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // create tensor
    prsm_tensor_t *ret = (out == NULL)
        ? prsm_tensor_create_shape(in->alloctr, in->ndim, in->shape)
        : out;

    // check size
    if (!prsm_tensor_match_shape(ret, in)) {
        prsm_tensor_resize_shape(ret, in->ndim, in->shape);
    }

    // copy over the values 
    prsm_tensor_assign(ret, in);

    // activate
    const size_t size = prsm_tensor_size(ret);
    VT_FOREACH(i, 0, size) ret->data[i] = prsm_math_prelu(ret->data[i], a);

    return ret;
}

// prsm_tensor_t *prsm_activate_prelu_d(prsm_tensor_t *out, const prsm_tensor_t *const in, const prsm_float c) {
//     return x >= 0 ? 1 : c;
// }

prsm_tensor_t *prsm_activate_softmax(prsm_tensor_t *out, const prsm_tensor_t *const in) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(in), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // create tensor
    prsm_tensor_t *ret = (out == NULL)
        ? prsm_tensor_create_shape(in->alloctr, in->ndim, in->shape)
        : out;

    // check size
    if (!prsm_tensor_match_shape(ret, in)) {
        prsm_tensor_resize_shape(ret, in->ndim, in->shape);
    }

    // copy over the values
    prsm_tensor_assign(ret, in);

    // normalize values
    prsm_tensor_apply_func(ret, PRSM_EXP);
    prsm_tensor_apply_scale_add(ret, 1/prsm_tensor_calc_sum(ret), 0);

    return ret;
}

prsm_tensor_t *prsm_activate_ssoftmax(prsm_tensor_t *out, const prsm_tensor_t *const in) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(in), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // create tensor
    prsm_tensor_t *ret = (out == NULL)
        ? prsm_tensor_create_shape(in->alloctr, in->ndim, in->shape)
        : out;

    // check size
    if (!prsm_tensor_match_shape(ret, in)) {
        prsm_tensor_resize_shape(ret, in->ndim, in->shape);
    }

    // copy over the values
    prsm_tensor_assign(ret, in);

    // normalize values
    prsm_tensor_apply_scale_add(ret, 1, -prsm_tensor_get_max(ret));
    prsm_tensor_apply_func(ret, PRSM_EXP);
    prsm_tensor_apply_scale_add(ret, 1/prsm_tensor_calc_sum(ret), 0);

    return ret;
}

prsm_tensor_t *prsm_activate_lsoftmax(prsm_tensor_t *out, const prsm_tensor_t *const in) {
    // check for invalid input
    VT_DEBUG_ASSERT(!prsm_tensor_is_null(in), "%s\n", prsm_status_to_str(PRSM_STATUS_ERROR_INVALID_ARGUMENTS));

    // create tensor
    prsm_tensor_t *ret = (out == NULL)
        ? prsm_tensor_create_shape(in->alloctr, in->ndim, in->shape)
        : out;

    // check size
    if (!prsm_tensor_match_shape(ret, in)) {
        prsm_tensor_resize_shape(ret, in->ndim, in->shape);
    }

    // copy over the values
    prsm_tensor_assign(ret, in);

    // calculate log(sum(exp(x - max)))
    prsm_float log_sum = 0;
    const prsm_float max = prsm_tensor_get_max(ret);
    const size_t size = prsm_tensor_size(ret);
    VT_FOREACH(i, 0, size)  {
        log_sum += PRSM_EXP(ret->data[i] - max);
    }
    log_sum = PRSM_LOG(log_sum);

    // shift values
    prsm_tensor_apply_scale_add(ret, 1, -max-log_sum);

    return ret;
}

