#include "main.h"
#include "perceptron.c"
#include "ann_mnist_digit_recognition.c"

static int test_num = 0;
#define TEST(func) { printf("(%d) ---> TESTING: %s\n", test_num, #func); func(); test_num++; }

void test_custom(void);
void test_tensor(void);
void test_math(void);
void test_activation(void);
void test_loss(void);
void test_layers(void);

int main(void) {
    vt_version_t 
        vt_v = vt_version_get(),
        prsm_v = prsm_version_get();
    printf("Vita (%s) | Prisma (%s)\n", vt_v.str, prsm_v.str);
    
    alloctr = vt_mallocator_create();
    {   
        // vt_debug_disable_output(true);

        TEST(test_custom);
        // TEST(test_tensor);
        // TEST(test_math);
        // TEST(test_activation);
        // TEST(test_loss);
        // TEST(test_layers);
    }
    vt_mallocator_print_stats(alloctr->stats);
    vt_mallocator_destroy(alloctr);
    return 0;
}

/* ------ TESTS ------ */

void test_custom(void) {
    // run_perceptron();
    run_ann_mnist_digit_recognition();
}

void test_tensor(void) {
    prsm_tensor_t *v0 = prsm_tensor_create_vec(alloctr, 5);

    // get/set
    prsm_tensor_set_all(v0, 1);
    prsm_tensor_set_val(v0, 0, 4);
    prsm_tensor_set_val(v0, 1, 0);
    assert(prsm_tensor_calc_sum(v0) == 7);
    assert(prsm_tensor_get_val(v0, 0) == 4);
    assert(prsm_tensor_get_max(v0) == 4);
    assert(prsm_tensor_get_max_index(v0) == 0);
    assert(prsm_tensor_get_min(v0) == 0);
    assert(prsm_tensor_get_min_index(v0) == 1);

    /*
     * VECTOR
     */

    // dup
    prsm_tensor_t *v1 = prsm_tensor_dup(v0);
    assert(prsm_tensor_equals(v0, v1));

    // add
    prsm_tensor_t *v2 = prsm_tensor_add(NULL, v0, v1);
    assert(prsm_tensor_calc_sum(v2) == 14);

    v2 = prsm_tensor_add(v2, v2, v1);
    assert(prsm_tensor_calc_sum(v2) == 21);

    // sub
    v2 = prsm_tensor_sub(v2, v2, v1);
    assert(prsm_tensor_calc_sum(v2) == 14);

    prsm_tensor_t *v3 = prsm_tensor_sub(NULL, v2, v1);
    assert(prsm_tensor_calc_sum(v3) == 7);

    // vector dot product
    prsm_float dotprod = prsm_tensor_vdot(v1, v2);
    assert(dotprod == 38);

    /*
     * MATRIX
     */

    // resize vector to matrix
    prsm_tensor_t *mat0 = prsm_tensor_dup(v3);
    prsm_tensor_set_zeros(mat0);
    prsm_tensor_resize(mat0, 2, 3, 3);
    prsm_tensor_set_identity(mat0);
    assert(prsm_tensor_calc_sum(mat0) == 3);

    // set/get
    prsm_tensor_set_val(mat0, vt_index_2d_to_1d(2, 0, mat0->shape[1]), 7);
    assert(prsm_tensor_get_val(mat0, vt_index_2d_to_1d(2, 0, mat0->shape[1])) == (float)7);

    // rand
    prsm_tensor_t *mat1 = prsm_tensor_create(alloctr, 2, 2, 3);
    prsm_tensor_rand(mat1);
    printf("RAND:\n"); prsm_tensor_display(mat1, NULL);
    prsm_tensor_rand_uniform(mat1, 0, 1);
    printf("RAND UNIFORM:\n"); prsm_tensor_display(mat1, NULL);
    VT_FOREACH(i, 0, prsm_tensor_size(mat1)) prsm_tensor_set_val(mat1, i, i);
    printf("INCREMENTAL:\n"); prsm_tensor_display(mat1, NULL);

    // multiplication
    prsm_tensor_t *mat2 = prsm_tensor_dot(NULL, mat1, mat0);
    assert(prsm_tensor_get_val(mat2, vt_index_2d_to_1d(0, 0, mat2->shape[1])) == 14);
    assert(prsm_tensor_get_val(mat2, vt_index_2d_to_1d(1, 0, mat2->shape[1])) == 38);

    // scale add
    prsm_tensor_apply_scale_add(mat2, 2, 1);
    assert(prsm_tensor_get_val(mat2, vt_index_2d_to_1d(0, 0, mat2->shape[1])) == 29);
    assert(prsm_tensor_get_val(mat2, vt_index_2d_to_1d(1, 0, mat2->shape[1])) == 77);

    // transpose
    const prsm_float beforeT_val = prsm_tensor_get_val(mat2, vt_index_2d_to_1d(0, 2, mat2->shape[1]));
    prsm_tensor_transpose(mat2);
    assert(prsm_tensor_get_val(mat2, vt_index_2d_to_1d(2, 0, mat2->shape[1])) == beforeT_val);

    // views
    assert(!prsm_tensor_is_view(mat2));
    
    prsm_tensor_t row_view_from_mat2 = prsm_tensor_make_view(mat2);
    row_view_from_mat2 = prsm_tensor_make_view_vec(mat2, 2);
    assert(prsm_tensor_get_val(&row_view_from_mat2, 0) == (prsm_float)5);

    row_view_from_mat2 = prsm_tensor_make_view_range(mat2, (size_t[]){0, 0, 1, 1});
    assert(prsm_tensor_get_val(&row_view_from_mat2, 3) == (prsm_float)9);

    // multiplication
    prsm_tensor_t 
        *mat_x = prsm_tensor_create_mat(alloctr, 4, 2),
        *mat_w = prsm_tensor_create_vec(alloctr, 2);
    VT_FOREACH(i, 0, prsm_tensor_size(mat_x)) prsm_tensor_set_val(mat_x, i, i);
    prsm_tensor_set_all(mat_w, 0.5);

    // mat by vec
    prsm_tensor_t *mat_mv = prsm_tensor_dot(NULL, mat_x, mat_w);
    assert(prsm_tensor_get_val(mat_mv, 0) == (prsm_float)0.5);
    assert(prsm_tensor_get_val(mat_mv, 1) == (prsm_float)2.5);
    assert(prsm_tensor_get_val(mat_mv, 2) == (prsm_float)4.5);
    assert(prsm_tensor_get_val(mat_mv, 3) == (prsm_float)6.5);
    
    // vec by mat
    prsm_tensor_transpose(mat_x);
    prsm_tensor_t *mat_vm = prsm_tensor_dot(NULL, mat_w, mat_x);
    assert(prsm_tensor_get_val(mat_vm, 0) == (prsm_float)0.5);
    assert(prsm_tensor_get_val(mat_vm, 1) == (prsm_float)2.5);
    assert(prsm_tensor_get_val(mat_vm, 2) == (prsm_float)4.5);
    assert(prsm_tensor_get_val(mat_vm, 3) == (prsm_float)6.5);

    // vec by vec
    prsm_tensor_t* __v1 = prsm_tensor_create(alloctr, 1, 2); __v1->data[0] = 1; __v1->data[1] = 2; 
    prsm_tensor_t* __v2 = prsm_tensor_create(alloctr, 1, 2); __v2->data[0] = 3; __v2->data[1] = 4; 
    prsm_tensor_t *__v3 = prsm_tensor_dot(NULL, __v1, __v2);
    prsm_tensor_display(__v3, NULL);
    assert(prsm_tensor_equals_array(__v3, (prsm_float[]){3, 6, 4, 8}, prsm_tensor_size(__v3)));

    // sum along axis
    prsm_tensor_t *sm = prsm_tensor_create_mat(alloctr, 2, 3);
    VT_FOREACH(i, 0, prsm_tensor_size(sm)) prsm_tensor_set_val(sm, i, i+1);

    // row-wise sum
    prsm_tensor_t *_sm = prsm_tensor_sum(NULL, sm, 0);
    assert(prsm_tensor_get_val(_sm, 0) == (prsm_float)5);
    assert(prsm_tensor_get_val(_sm, 1) == (prsm_float)7);
    assert(prsm_tensor_get_val(_sm, 2) == (prsm_float)9);

    // col-wise sum
    prsm_tensor_sum(_sm, sm, 1);
    assert(prsm_tensor_get_val(_sm, 0) == (prsm_float)6);
    assert(prsm_tensor_get_val(_sm, 1) == (prsm_float)15);

    // element-wise multiplication
    prsm_tensor_t *elm1 = prsm_tensor_create_mat(alloctr, 2, 2);
    prsm_tensor_assign_array(elm1, (prsm_float[]) {
        10, 2, 40, 4
    }, prsm_tensor_size(elm1));
    
    prsm_tensor_t *elm2 = prsm_tensor_create_mat(alloctr, 2, 2);
    prsm_tensor_assign_array(elm1, (prsm_float[]) {
        1, 2, 3, 4
    }, prsm_tensor_size(elm1));
    
    prsm_tensor_t *elm_expected = prsm_tensor_create_mat(alloctr, 2, 2);
    prsm_tensor_assign_array(elm1, (prsm_float[]) {
        10, 4, 120, 16
    }, prsm_tensor_size(elm1));

    prsm_tensor_t *elm3 = prsm_tensor_mul(NULL, elm1, elm2);
    assert(prsm_tensor_equals(elm3, elm_expected));

    // 3D summation
    prsm_tensor_t *nd3m = prsm_tensor_create(alloctr, 3, 3, 2, 2);
    prsm_tensor_assign_array(nd3m, (prsm_float[]) {
        1, 2,
        3, 4,
        
        5, 6,
        7, 8,

        9, 0,
        1, 2
    }, prsm_tensor_size(nd3m));

    // expected output
    prsm_tensor_t *nd3m_msum_expected = prsm_tensor_create_mat(alloctr, 2, 2);
    prsm_tensor_assign_array(nd3m_msum_expected, (prsm_float[]) {
        15, 8, 
        11, 14
    }, prsm_tensor_size(nd3m_msum_expected));
    prsm_tensor_t *nd3m_rsum_expected = prsm_tensor_create_mat(alloctr, 3, 2);
    prsm_tensor_assign_array(nd3m_rsum_expected, (prsm_float[]) {
        4, 6,
        12, 14, 
        10, 2
    }, prsm_tensor_size(nd3m_rsum_expected));

    // z-wise sum
    prsm_tensor_t *nd3m_sum = prsm_tensor_sum(NULL, nd3m, 0);
    assert(prsm_tensor_equals(nd3m_sum, nd3m_msum_expected));

    // row-wise sum for 3D matrix
    prsm_tensor_sum(nd3m_sum, nd3m, 1);
    assert(prsm_tensor_equals(nd3m_sum, nd3m_rsum_expected));

    // col-wise sum for 3D matrix
    prsm_tensor_sum(nd3m_sum, nd3m, 2);

    // flatten
    prsm_tensor_flatten(nd3m_sum);
    assert(prsm_tensor_equals_array(nd3m_sum, (prsm_float[]){3, 7, 11, 15, 9, 3}, prsm_tensor_size(nd3m_sum)));

    // diagflat
    prsm_tensor_diagflat(nd3m_sum);
    assert(prsm_tensor_equals_array(nd3m_sum, (prsm_float[]){
        3, 0,  0,  0, 0, 0, 
        0, 7,  0,  0, 0, 0,
        0, 0, 11,  0, 0, 0,
        0, 0,  0, 15, 0, 0,
        0, 0,  0,  0, 9, 0, 
        0, 0,  0,  0, 0, 3
    }, prsm_tensor_size(nd3m_sum)));
}

void test_math(void) {
    prsm_tensor_t *m0 = prsm_tensor_create_mat(alloctr, 3, 3);

    VT_FOREACH(i, 0, prsm_tensor_size(m0)) prsm_tensor_set_val(m0, i, i);
    prsm_tensor_apply_func(m0, prsm_math_sigmoid);
    assert((int32_t)(prsm_tensor_get_val(m0, 0)*100) == 50);
    assert((int32_t)(prsm_tensor_get_val(m0, 1)*100) == 73);
    assert((int32_t)(prsm_tensor_get_val(m0, 2)*100) == 88);

    VT_FOREACH(i, 0, prsm_tensor_size(m0)) prsm_tensor_set_val(m0, i, i);
    prsm_tensor_apply_func(m0, prsm_math_sigmoid_d);
    assert((int32_t)(prsm_tensor_get_val(m0, 0)*100) == 25);
    assert((int32_t)(prsm_tensor_get_val(m0, 1)*100) == 19);
    assert((int32_t)(prsm_tensor_get_val(m0, 2)*100) == 10);
}

void test_activation(void) {
    prsm_tensor_t *data = prsm_tensor_create_vec(alloctr, 4);
    prsm_tensor_t *expected_output = prsm_tensor_create_vec(alloctr, 4);
    prsm_tensor_t *d_expected_output = prsm_tensor_create_vec(alloctr, 4);
    prsm_tensor_assign_array(data, (prsm_float[]) {
        10, 2, 40, 4
    }, prsm_tensor_size(data));
    prsm_tensor_assign_array(expected_output, (prsm_float[]) {
        0, 0, 1, 0
    }, prsm_tensor_size(expected_output));
    prsm_tensor_assign_array(d_expected_output, (prsm_float[]) {
        0.1442, 0.1442, 0.2493, 0.1442
    }, prsm_tensor_size(d_expected_output));

    // softmax
    prsm_tensor_t *output = prsm_activate_softmax(NULL, data);
    prsm_tensor_apply_func(output, PRSM_ROUND);
    assert(prsm_tensor_equals(output, expected_output));

    // d'softmax
    prsm_tensor_t *d_output = prsm_activate_softmax_d(NULL, expected_output);
    assert(prsm_tensor_equals_approx(d_output, d_expected_output, 0.001));

    // stable softmax
    output = prsm_activate_ssoftmax(output, data);
    prsm_tensor_apply_func(output, PRSM_ROUND);
    assert(prsm_tensor_equals(output, expected_output));

    // d'stable softmax
    d_output = prsm_activate_ssoftmax_d(d_output, expected_output);
    assert(prsm_tensor_equals_approx(d_output, d_expected_output, 0.001));

    // log softmax
    prsm_tensor_resize(data, 1, 3);
    prsm_tensor_assign_array(data, (prsm_float[]) {
        1, 2, 3
    }, prsm_tensor_size(data));

    prsm_tensor_resize(expected_output, 1, 3);
    prsm_tensor_assign_array(expected_output, (prsm_float[]) {
        -2.4076061 , -1.4076061 , -0.40760612
    }, prsm_tensor_size(expected_output));

    prsm_tensor_resize(d_expected_output, 1, 3);
    prsm_tensor_assign_array(d_expected_output, (prsm_float[]) {
        0.9099, 0.7552, 0.3347
    }, prsm_tensor_size(d_expected_output));

    output = prsm_activate_lsoftmax(output, data);
    assert(prsm_tensor_equals_approx(output, expected_output, 0.001));

    // d'log softmax
    d_output = prsm_activate_lsoftmax_d(d_output, data);
    assert(prsm_tensor_equals_approx(d_output, d_expected_output, 0.001));
}

void test_loss(void) {
    prsm_tensor_t *y = prsm_tensor_create_vec(alloctr, 4);
    VT_FOREACH(i, 0, prsm_tensor_size(y)) prsm_tensor_set_val(y, i, i);
    
    prsm_tensor_t *yhat = prsm_tensor_create_vec(alloctr, 4);
    VT_FOREACH(i, 0, prsm_tensor_size(yhat)) prsm_tensor_set_val(yhat, i, i+0.5);

    // MAE, MSE
    assert(prsm_loss_mae(yhat, y) == (prsm_float)0.5);
    assert(prsm_loss_mse(yhat, y) == (prsm_float)0.25);
    assert(prsm_loss_rmse(yhat, y) == (prsm_float)0.5);

    // MAE derivative (gradients)
    prsm_tensor_t *grad_expected = prsm_tensor_dup(yhat);
    prsm_tensor_assign_array(grad_expected, (prsm_float[]) {
        0.25, 0.25, 0.25, 0.25
    }, prsm_tensor_size(grad_expected));

    prsm_tensor_t *grad = prsm_loss_mae_d(NULL, yhat, y);
    assert(prsm_tensor_equals(grad_expected, grad));

    // MSE derivative (gradients)
    grad = prsm_loss_mse_d(grad, yhat, y);
    assert(prsm_tensor_equals(grad_expected, grad));

    // MSE derivative (gradients)
    grad = prsm_loss_rmse_d(grad, yhat, y);
    assert(prsm_tensor_equals(grad_expected, grad));

    // Binary cross entropy
    prsm_tensor_resize(y, 2, 3, 6);
    prsm_tensor_resize(yhat, 2, 3, 6);
    prsm_tensor_assign_array(y, (prsm_float[]) {
        1, 0, 1, 1, 0, 0, 
        1, 1, 0, 1, 0, 0,
        1, 0, 1, 1, 0, 0
    }, prsm_tensor_size(yhat));
    prsm_tensor_assign_array(yhat, (prsm_float[]) {
        0.2, 0.1, 0.8, 0.7, 0.1, 0.2, 
        0.3, 0.4, 0.5, 0.6, 0.7, 0.7,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6
    }, prsm_tensor_size(yhat));

    prsm_tensor_t *loss_results_target = prsm_tensor_create_vec(alloctr, prsm_tensor_shape(y)[0]);
    prsm_tensor_assign_array(loss_results_target, (prsm_float[]) {
        0.4371866, 0.95536333, 1.0425713
    }, prsm_tensor_size(loss_results_target));

    prsm_tensor_t *bce_grads_expected = prsm_tensor_dup(yhat);
    prsm_tensor_assign_array(bce_grads_expected, (prsm_float[]) {
        -0.8333,  0.1852, -0.2083, -0.2381,  0.1852,  0.2083,
        -0.5556, -0.4167,  0.3333, -0.2778,  0.5556,  0.5556,
        -1.6667,  0.2083, -0.5556, -0.4167,  0.3333,  0.4167
    }, prsm_tensor_size(bce_grads_expected));

    VT_FOREACH(i, 0, prsm_tensor_shape(y)[0]) {
        const prsm_tensor_t y_ = prsm_tensor_make_view_vec(y, i);
        const prsm_tensor_t yhat_ = prsm_tensor_make_view_vec(yhat, i);
        const prsm_tensor_t loss_grads_ = prsm_tensor_make_view_vec(bce_grads_expected, i);

        // loss
        const prsm_float loss = prsm_loss_bce(&yhat_, &y_);
        assert(vt_math_is_close(loss, prsm_tensor_get_val(loss_results_target, i), 0.001));

        // loss grad
        grad = prsm_loss_bce_d(grad, &yhat_, &y_);
        assert(prsm_tensor_equals_approx(grad, &loss_grads_, 0.001));
    }

    // Categorical cross entropy
    prsm_tensor_assign_array(loss_results_target, (prsm_float[]) {
        4.415068, 6.1205416, 6.64866
    }, prsm_tensor_size(loss_results_target));

    prsm_tensor_t *cce_grads_expected = prsm_tensor_dup(yhat);
    prsm_tensor_assign_array(cce_grads_expected, (prsm_float[]) {
        -3.5714,  1.4285,  0.1785,  0.0000,  1.4285,  1.4285,
        -2.3958, -1.5625,  0.9375, -0.7291,  0.9375,  0.9375,
        -8.5714,  1.4285, -1.9047, -1.0714,  1.4285,  1.4285
    }, prsm_tensor_size(cce_grads_expected));

    VT_FOREACH(i, 0, prsm_tensor_shape(y)[0]) {
        const prsm_tensor_t y_ = prsm_tensor_make_view_vec(y, i);
        const prsm_tensor_t yhat_ = prsm_tensor_make_view_vec(yhat, i);
        const prsm_tensor_t loss_grads_ = prsm_tensor_make_view_vec(cce_grads_expected, i);

        // loss
        const prsm_float loss = prsm_loss_cce(&yhat_, &y_);
        assert(vt_math_is_close(loss, prsm_tensor_get_val(loss_results_target, i), 0.001));

        // loss grad
        grad = prsm_loss_cce_d(grad, &yhat_, &y_);
        assert(prsm_tensor_equals_approx(grad, &loss_grads_, 0.001));
    }
}

void test_layers(void) {
    //
}




