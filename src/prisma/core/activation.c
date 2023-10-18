#include "prisma/core/activation.h"

prsm_float prsm_activation_sigmoid(const prsm_float x) {
    return 1 / (1 + PRSM_EXP(-1 * x));
}

prsm_float prsm_activation_sigmoid_d(const prsm_float x) {
    const prsm_float s = prsm_activation_sigmoid(x);
    return s * (1 - s);
}

prsm_float prsm_activation_tanh(const prsm_float x) {
    return PRSM_TANH(x);
}

prsm_float prsm_activation_tahn_d(const prsm_float x) {
    const prsm_float s = prsm_activation_tanh(x);
    return 1 - (s * s);
}

// NOTE: TODO: move to tensor
// takes in vector (array): SF(vec) => out = exp(vec); out/sum(out);
// takes in vector (array): SSF(vec) => shift = vec - max(vec); out = exp(shift); out/sum(out);
// prsm_float prsm_activation_softmax(const prsm_float x);
// prsm_float prsm_activation_softmax_d(const prsm_float x);
// prsm_float prsm_activation_ssoftmax(const prsm_float x);
// prsm_float prsm_activation_ssoftmax_d(const prsm_float x);

prsm_float prsm_activation_linear(const prsm_float x) {
    return x;
}

prsm_float prsm_activation_linear_d(const prsm_float x) {
    (void)x;
    return 1;
}

prsm_float prsm_activation_ramp(prsm_float x) {
    return x * (x > 0) + 0.1 * x;
}

prsm_float prsm_activation_ramp_d(prsm_float x) {
    return (x > 0) + 0.1;
}

prsm_float prsm_activation_htanh(prsm_float x) {
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}

prsm_float prsm_activation_htanh_d(prsm_float x) {
    if (x > -1 && x < 1) return 1;
    return 0;
}

prsm_float prsm_activation_relu(const prsm_float x) {
    return PRSM_MAX(0, x);
}

prsm_float prsm_activation_relu_d(const prsm_float x) {
    return x >= 0 ? 1 : 0;
}

prsm_float prsm_activation_lrelu(const prsm_float x) {
    return PRSM_MAX(0, x) + 0.01 * PRSM_MIN(0, x);
}

prsm_float prsm_activation_lrelu_d(const prsm_float x) {
    return x >= 0 ? 1 : 0.01;
}

prsm_float prsm_activation_elu(const prsm_float x) {
    return x > 0 ? x : (PRSM_EXP(x) - 1);
}

prsm_float prsm_activation_elu_d(const prsm_float x) {
    return x > 0 ? 1 : PRSM_EXP(x);
}

prsm_float prsm_activation_selu(const prsm_float x) {
    return x > 0 ? (1.0507*x) : (1.0507 * 1.6732 * (PRSM_EXP(x) - 1));
}

prsm_float prsm_activation_selu_d(const prsm_float x) {
    return x > 0 ? 1.0507 : (1.0507 * 1.6732 * PRSM_EXP(x));
}

prsm_float prsm_activation_prelu(const prsm_float x, const prsm_float c) {
    return PRSM_MAX(0, x) + c * PRSM_MIN(0, x);
}

prsm_float prsm_activation_prelu_d(const prsm_float x, const prsm_float c) {
    return x >= 0 ? 1 : c;
}

