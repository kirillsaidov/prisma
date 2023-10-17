#include "prisma/core/activation.h"

prsm_float prsm_activation_sigmoid(const prsm_float v) {
    return 1 / (1 + PRSM_EXP(-1 * v));
}

prsm_float prsm_activation_sigmoid_d(const prsm_float v) {
    const prsm_float s = prsm_activation_sigmoid(v);
    return s * (1 - s);
}

prsm_float prsm_activation_tanh(const prsm_float v) {
    return PRSM_TANH(v);
}

prsm_float prsm_activation_tahn_d(const prsm_float v) {
    const prsm_float s = prsm_activation_tanh(v);
    return 1 - (s * s);
}

// NOTE: TODO: move to tensor
// takes in vector (array): SF(vec) => out = exp(vec); out/sum(out);
// takes in vector (array): SSF(vec) => shift = vec - max(vec); out = exp(shift); out/sum(out);
// prsm_float prsm_activation_softmax(const prsm_float v);
// prsm_float prsm_activation_softmax_d(const prsm_float v);
// prsm_float prsm_activation_ssoftmax(const prsm_float v);
// prsm_float prsm_activation_ssoftmax_d(const prsm_float v);

prsm_float prsm_activation_linear(const prsm_float v) {
    return v;
}

prsm_float prsm_activation_linear_d(const prsm_float v) {
    (void)v;
    return 1;
}

prsm_float prsm_activation_relu(const prsm_float v) {
    return PRSM_MAX(0, v);
}

prsm_float prsm_activation_relu_d(const prsm_float v) {
    return v >= 0 ? 1 : 0;
}

prsm_float prsm_activation_lrelu(const prsm_float v) {
    return PRSM_MAX(0, v) + 0.01 * PRSM_MIN(0, v);
}

prsm_float prsm_activation_lrelu_d(const prsm_float v) {
    return v >= 0 ? 1 : 0.01;
}

prsm_float prsm_activation_prelu(const prsm_float v, const prsm_float c) {
    return PRSM_MAX(0, v) + c * PRSM_MIN(0, v);
}

prsm_float prsm_activation_prelu_d(const prsm_float v, const prsm_float c) {
    return v >= 0 ? 1 : c;
}

prsm_float prsm_activation_elu(const prsm_float v) {
    return v > 0 ? v : (PRSM_EXP(v) - 1);
}

prsm_float prsm_activation_elu_d(const prsm_float v) {
    return v > 0 ? 1 : PRSM_EXP(v);
}

prsm_float prsm_activation_selu(const prsm_float v, const prsm_float s, const prsm_float a) {
    return v > 0 ? (s > 0 ? s : 1.0507)*v : (s > 0 ? s : 1.0507)*(a > 0 ? a : 1.6732)*(PRSM_EXP(v) - 1);
}

prsm_float prsm_activation_selu_d(const prsm_float v, const prsm_float s, const prsm_float a) {
    return v > 0 ? (s > 0 ? s : 1.0507) : (s > 0 ? s : 1.0507) * (a > 0 ? a : 1.6732) * PRSM_EXP(v);
}

