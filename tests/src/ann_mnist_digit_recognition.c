#include "main.h"
#include <curl/curl.h>

#define MNIST_TRAIN "mnist_train.csv"
#define MNIST_TEST "mnist_test.csv"

void ann_download_csv(const char *const url, const char *const filepath);
void ann_read_mnist_data(const char *const file, prsm_tensor_t *x, prsm_tensor_t *y);
vt_vec_t *ann_model_init_params(const size_t n_x, const size_t n_h, const size_t n_y);
vt_vec_t *ann_model_forward(prsm_tensor_t *x, vt_vec_t *params, vt_vec_t *forward_cache);
vt_vec_t *ann_model_backward(prsm_tensor_t *x, prsm_tensor_t *y, vt_vec_t *params, vt_vec_t *forward_cache, vt_vec_t *backward_cache);
vt_vec_t *ann_model_update(vt_vec_t *params, vt_vec_t *backward_cache, const prsm_float learning_rate);

void run_ann_mnist_digit_recognition(void) {
    vt_file_write("debug.log", "");
    vt_debug_redirect_output("debug.log");

    VT_LOG_INFO("Looking for MNIST digits dataset...");
    if (!vt_path_exists(CACHE_FOLDER)) {
        VT_LOG_INFO("Creating a cache folder...");
        vt_path_mkdir("assets/cache");
    }

    if (!vt_path_exists(CACHE_FOLDER MNIST_TRAIN)) {
        VT_LOG_INFO("Downloading %s dataset...", MNIST_TRAIN);
        ann_download_csv("https://pjreddie.com/media/files/mnist_train.csv", CACHE_FOLDER MNIST_TRAIN);
    } else {
        VT_LOG_INFO("MNIST digits train dataset found.");
    }

    if (!vt_path_exists(CACHE_FOLDER MNIST_TEST)) {
        VT_LOG_INFO("Downloading %s dataset...", MNIST_TEST);
        ann_download_csv("https://pjreddie.com/media/files/mnist_test.csv", CACHE_FOLDER MNIST_TEST);
    } else {
        VT_LOG_INFO("MNIST digits test dataset found.");
    }

    VT_LOG_INFO("Creating tensors...");
    const size_t num_rows = 100;
    const size_t num_features = 28*28;
    prsm_tensor_t *x_train, *y_train, *x_test, *y_test;
    x_train = prsm_tensor_create_mat(alloctr, num_rows, num_features);
    x_test = prsm_tensor_create_mat(alloctr, num_rows, num_features);
    y_train = prsm_tensor_create_mat(alloctr, num_rows, 10);
    y_test = prsm_tensor_create_mat(alloctr, num_rows, 10);

    VT_LOG_INFO("Loading data into memory: %zu instances", num_rows);
    ann_read_mnist_data(CACHE_FOLDER MNIST_TRAIN, x_train, y_train);
    ann_read_mnist_data(CACHE_FOLDER MNIST_TEST, x_test, y_test);

    VT_LOG_INFO("Normalize the data...");
    prsm_tensor_apply_scale_add(x_train, 1.0/255.0, 0);
    prsm_tensor_apply_scale_add(x_test, 1.0/255.0, 0);

    VT_LOG_INFO("Initializing parameters...");
    const size_t layer_output_size = 10;
    const size_t layer_hidden_size = 100;
    vt_vec_t *params = ann_model_init_params(num_features, layer_hidden_size, layer_output_size);

    VT_LOG_INFO("Initializing model options...");
    const size_t epochs = 100;
    const prsm_float alpha = 0.05;

    VT_LOG_INFO("\talpha         = %.2f", alpha);
    VT_LOG_INFO("\tactivation l1 = %s", VT_STRING_OF(prsm_math_sigmoid));
    VT_LOG_INFO("\tactivation l2 = %s", VT_STRING_OF(prsm_activate_softmax));
    VT_LOG_INFO("\tloss          = %s", VT_STRING_OF(prsm_loss_cce));
    VT_LOG_INFO("\tepochs        = %zu", epochs);

    VT_LOG_INFO("Start training...");
    vt_vec_t *forward_cache = vt_vec_create(10, sizeof(h_keyval_t), alloctr);
    vt_vec_t *backward_cache = vt_vec_create(10, sizeof(h_keyval_t), alloctr);
    VT_FOREACH(epoch, 0, epochs) {
        /* -----------------------
        * FORWARD
        */
        forward_cache = ann_model_forward(x_train, params, forward_cache);

        /* -----------------------
        * COST
        */
        prsm_tensor_t *yhat = h_find_val(forward_cache, "a2");
        const prsm_float cost = prsm_loss_cce(yhat, y_train);
        VT_LOG_INFO("\tEpoch %3zu | Error: %.2f", epoch, cost);

        /* -----------------------
        * BACKWARD
        */
        backward_cache = ann_model_backward(x_train, y_train, params, forward_cache, backward_cache);
        // params = ann_model_update(params, backward_cache, alpha);
    }

    // VT_FOREACH(k, 0, 5) {
    //     // y (label) value
    //     prsm_tensor_t yt = prsm_tensor_make_view_vec(y_train, k);
    //     prsm_tensor_display(&yt, NULL);

    //     // x value
    //     prsm_tensor_t t = prsm_tensor_make_view_vec(x_train, k);
    //     VT_FOREACH(i, 0, 28) {
    //         VT_FOREACH(j, 0, 28) {
    //             const prsm_float value = t.data[vt_index_2d_to_1d(i, j, 28)];
    //             printf("%d", value == 0 ? 0 : 1);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
}

void ann_download_csv(const char *const url, const char *const filepath) {
    FILE *fp = fopen(filepath, "wb");

    CURL *handle = curl_easy_init();
    curl_easy_setopt(handle, CURLOPT_URL, url);
    curl_easy_setopt(handle, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(handle, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(handle, CURLOPT_USERAGENT, "curl/7.35.0");
    curl_easy_setopt(handle, CURLOPT_MAXREDIRS, 50L);
    curl_easy_setopt(handle, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(handle, CURLOPT_FAILONERROR, 1L);

    CURLcode ret = curl_easy_perform(handle);
    if (ret != CURLE_OK) VT_LOG_FATAL("Failed to download <%s> => %s", url, curl_easy_strerror(ret));

    curl_easy_cleanup(handle);
    fclose(fp);

    handle = NULL;
    fp = NULL;
}

void ann_read_mnist_data(const char *const file, prsm_tensor_t *x, prsm_tensor_t *y) {
    char buffer[2048];
    size_t current_row = 0;
    const size_t read_rows = prsm_tensor_shape(x)[0];
    const size_t read_cols = prsm_tensor_shape(x)[1];
    vt_plist_t *line_items = vt_plist_create(read_cols, alloctr);

    /**
     * READING TRAIN DATA
     */
    VT_LOG_INFO("Reading %s", file);
    FILE *fp = fopen(file, "r"); 
    {
        while(current_row < read_rows && fgets(buffer, sizeof(buffer), fp)) {
            // split
            vt_str_t line = vt_str_create_static(buffer);

            // ----------------
            line_items = vt_str_split(line_items, &line, ",");
            VT_FOREACH(i, 0, vt_plist_len(line_items)) {
                // get item
                vt_str_t *item = (vt_str_t*)vt_plist_get(line_items, i);
                vt_str_strip(item);

                // convert to prsm_float
                const prsm_float value = vt_conv_str_to_f(vt_str_z(item));
                
                // save values
                if (i == 0) prsm_tensor_set_val(y, vt_index_2d_to_1d(current_row, value, 10), 1);
                else prsm_tensor_set_val(x, vt_index_2d_to_1d(current_row, i-1, read_cols), value);
            }
            // ----------------

            // update
            current_row++;
        }
    } 
    fclose(fp);
}

vt_vec_t *ann_model_init_params(const size_t n_x, const size_t n_h, const size_t n_y) {
    VT_LOG_INFO("\tCreating weights and biases...");
    prsm_tensor_t *w1, *w2, *b1, *b2;
    w1 = prsm_tensor_create_mat(alloctr, n_h, n_x);
    b1 = prsm_tensor_create_vec(alloctr, n_h);
    w2 = prsm_tensor_create_mat(alloctr, n_y, n_h);
    b2 = prsm_tensor_create_vec(alloctr, n_y);

    VT_LOG_INFO("\tNN created: (%zu, %zu, %zu)", n_x, n_h, n_y);
    VT_LOG_INFO("\tRandomizing weights...");
    prsm_tensor_rand(w1);
    prsm_tensor_rand(w2);
    prsm_tensor_set_all(b1, 1);
    prsm_tensor_set_all(b2, 1);

    vt_vec_t *params = vt_vec_create(10, sizeof(h_keyval_t), alloctr);
    vt_vec_push(params, &(h_keyval_t){.key = "w1", .value=w1});
    vt_vec_push(params, &(h_keyval_t){.key = "b1", .value=b1});
    vt_vec_push(params, &(h_keyval_t){.key = "w2", .value=w2});
    vt_vec_push(params, &(h_keyval_t){.key = "b2", .value=b2});

    return params;
}

vt_vec_t *ann_model_forward(prsm_tensor_t *x, vt_vec_t *params, vt_vec_t *forward_cache) {
    // free forward_cache
    const size_t fclen = vt_vec_len(forward_cache);
    VT_FOREACH(i, 0, fclen) {
        h_keyval_t *keyval = vt_vec_get(forward_cache, i);
        prsm_tensor_destroy(keyval->value);
    }
    vt_vec_clear(forward_cache);

    // get params
    prsm_tensor_t *w1, *w2, *b1, *b2;
    w1 = h_find_val(params, "w1");
    b1 = h_find_val(params, "b1");
    w2 = h_find_val(params, "w2");
    b2 = h_find_val(params, "b2");

    /**
     * LAYER 1
     */

    // z1 = x * w1_T + b1
    if (prsm_tensor_shape(x)[1] != prsm_tensor_shape(w1)[0]) prsm_tensor_transpose(w1);
    prsm_tensor_t *z1 = prsm_tensor_mul(NULL, x, w1);

    // add bias
    VT_FOREACH(i, 0, prsm_tensor_shape(z1)[0]) {
        prsm_tensor_t t = prsm_tensor_make_view_vec(z1, i);
        prsm_tensor_add(&t, &t, b1);
    }

    // activate: a1 = relu(z1)
    prsm_tensor_t *a1 = prsm_tensor_dup(z1);
    prsm_activate_relu(a1, a1);

    // printf("shape of  x: (%zu, %zu)\n", prsm_tensor_shape(x)[0], prsm_tensor_shape(x)[1]);
    // printf("shape of w1: (%zu, %zu)\n", prsm_tensor_shape(w1)[0], prsm_tensor_shape(w1)[1]);
    // printf("shape of z1: (%zu, %zu)\n", prsm_tensor_shape(z1)[0], prsm_tensor_shape(z1)[1]);

    /**
     * LAYER 2
     */

    // z2 = z1 * w2_T + b2
    if (prsm_tensor_shape(z1)[1] != prsm_tensor_shape(w2)[0]) prsm_tensor_transpose(w2);
    prsm_tensor_t *z2 = prsm_tensor_mul(NULL, z1, w2);

    // add bias
    VT_FOREACH(i, 0, prsm_tensor_shape(z2)[0]) {
        prsm_tensor_t t = prsm_tensor_make_view_vec(z2, i);
        prsm_tensor_add(&t, &t, b2);
    }

    // activate: a2 = softmax(z2)
    prsm_tensor_t *a2 = prsm_tensor_dup(z2);
    VT_FOREACH(i, 0, prsm_tensor_shape(a2)[0]) {
        prsm_tensor_t t = prsm_tensor_make_view_vec(a2, i);
        prsm_activate_ssoftmax(&t, &t);
    }

    // printf("shape of z1: (%zu, %zu)\n", prsm_tensor_shape(z1)[0], prsm_tensor_shape(z1)[1]);
    // printf("shape of w2: (%zu, %zu)\n", prsm_tensor_shape(w2)[0], prsm_tensor_shape(w2)[1]);
    // printf("shape of z2: (%zu, %zu)\n", prsm_tensor_shape(z2)[0], prsm_tensor_shape(z2)[1]);
    // printf("shape of a2: (%zu, %zu)\n", prsm_tensor_shape(a2)[0], prsm_tensor_shape(a2)[1]);

    // update forward_cache
    vt_vec_push(forward_cache, &(h_keyval_t){.key = "z1", .value=z1});
    vt_vec_push(forward_cache, &(h_keyval_t){.key = "a1", .value=a1});
    vt_vec_push(forward_cache, &(h_keyval_t){.key = "z2", .value=z2});
    vt_vec_push(forward_cache, &(h_keyval_t){.key = "a2", .value=a2});

    return forward_cache;
}

vt_vec_t *ann_model_backward(prsm_tensor_t *x, prsm_tensor_t *y, vt_vec_t *params, vt_vec_t *forward_cache, vt_vec_t *backward_cache) {
    // free backward_cache
    const size_t bclen = vt_vec_len(backward_cache);
    VT_FOREACH(i, 0, bclen) {
        h_keyval_t *keyval = vt_vec_get(backward_cache, i);
        prsm_tensor_destroy(keyval->value);
    }
    vt_vec_clear(backward_cache);

    // get params and forward_cache
    prsm_tensor_t *w1, *w2, *b1, *b2, *a1, *a2;
    w1 = h_find_val(params, "w1");
    b1 = h_find_val(params, "b1");
    w2 = h_find_val(params, "w2");
    b2 = h_find_val(params, "b2");
    a1 = h_find_val(forward_cache, "a1");
    a2 = h_find_val(forward_cache, "a2");

    // number of observations
    const float m = prsm_tensor_shape(x)[0];

    /**
     * LAYER 2
     */
    
    // dz2 = a2 - y
    prsm_tensor_t *dz2 = prsm_tensor_sub(NULL, a2, y);

    // dw2 = 1/m * dz2 * a1
    printf("here\n");
    printf("shape of dz2: (%zu, %zu)\n", prsm_tensor_shape(dz2)[0], prsm_tensor_shape(dz2)[1]);
    printf("shape of  a1: (%zu, %zu)\n", prsm_tensor_shape(a1)[0], prsm_tensor_shape(a1)[1]);
    prsm_tensor_transpose(dz2);
    prsm_tensor_t *dw2 = prsm_tensor_mul(NULL, dz2, a1);
    prsm_tensor_apply_scale_add(dw2, 1.0/m, 0);
    printf("here: done\n");

    // db2 = 1/m * sum(dz2, 1)
    prsm_tensor_t *db2 = prsm_tensor_sum(NULL, dz2, 1);
    prsm_tensor_apply_scale_add(db2, 1.0/m, 0);

    // printf("shape of dz2: (%zu, %zu)\n", prsm_tensor_shape(dz2)[0], prsm_tensor_shape(dz2)[1]);
    // printf("shape of  a1: (%zu, %zu)\n", prsm_tensor_shape(a1)[0], prsm_tensor_shape(a1)[1]);
    // printf("shape of dw2: (%zu, %zu)\n", prsm_tensor_shape(dw2)[0], prsm_tensor_shape(dw2)[1]);
    // printf("shape of db2: (%zu, 1)\n", prsm_tensor_shape(db2)[0]);

    /**
     * LAYER 1
     */
    
    // dz1 = w2 * dz2 * acrivation_derivative(a1)
    prsm_tensor_t *dz1_tmp1 = prsm_tensor_mul(NULL, w2, dz2); 
    prsm_tensor_t *dz1_tmp2 = prsm_activate_relu_d(NULL, a1);
    prsm_tensor_transpose(dz1_tmp2);
    prsm_tensor_t *dz1 = prsm_tensor_mul_elwise(NULL, dz1_tmp1, dz1_tmp2);
    prsm_tensor_destroy(dz1_tmp1);
    prsm_tensor_destroy(dz1_tmp2);

    // dw1 = 1/m * dz1 * x
    prsm_tensor_t *dw1 = prsm_tensor_mul(NULL, dz1, x);
    prsm_tensor_apply_scale_add(dw1, 1.0/m, 0);

    // db1 = 1/m * sum(dz1, 1)
    prsm_tensor_t *db1 = prsm_tensor_sum(NULL, dz1, 1);
    prsm_tensor_apply_scale_add(db1, 1.0/m, 0);

    // printf("shape of  w2: (%zu, %zu)\n", prsm_tensor_shape(w2)[0], prsm_tensor_shape(w2)[1]);
    // printf("shape of dz2: (%zu, %zu)\n", prsm_tensor_shape(dz2)[0], prsm_tensor_shape(dz2)[1]);
    // // printf("shape of dz1_tmp1: (%zu, %zu)\n", prsm_tensor_shape(dz1_tmp1)[0], prsm_tensor_shape(dz1_tmp1)[1]);
    // // printf("shape of dz1_tmp2: (%zu, %zu)\n", prsm_tensor_shape(dz1_tmp2)[0], prsm_tensor_shape(dz1_tmp2)[1]);
    // printf("shape of dz1: (%zu, %zu)\n", prsm_tensor_shape(dz1)[0], prsm_tensor_shape(dz1)[1]);
    // printf("shape of   x: (%zu, %zu)\n", prsm_tensor_shape(x)[0], prsm_tensor_shape(x)[1]);
    // printf("shape of dw1: (%zu, %zu)\n", prsm_tensor_shape(dw1)[0], prsm_tensor_shape(dw1)[1]);
    // printf("shape of db1: (%zu, 1)\n", prsm_tensor_shape(db1)[0]);

    // update backward_cache
    vt_vec_push(backward_cache, &(h_keyval_t){.key = "dw1", .value=dw1});
    vt_vec_push(backward_cache, &(h_keyval_t){.key = "db1", .value=db1});
    vt_vec_push(backward_cache, &(h_keyval_t){.key = "dw2", .value=dw2});
    vt_vec_push(backward_cache, &(h_keyval_t){.key = "db2", .value=db2});

    return backward_cache;
}

vt_vec_t *ann_model_update(vt_vec_t *params, vt_vec_t *backward_cache, const prsm_float learning_rate) {
    // get params
    prsm_tensor_t *w1, *w2, *b1, *b2;
    w1 = h_find_val(params, "w1");
    b1 = h_find_val(params, "b1");
    w2 = h_find_val(params, "w2");
    b2 = h_find_val(params, "b2");

    // get backward_cache (gradients)
    prsm_tensor_t *dw1, *dw2, *db1, *db2;
    dw1 = h_find_val(backward_cache, "dw1");
    db1 = h_find_val(backward_cache, "db1");
    dw2 = h_find_val(backward_cache, "dw2");
    db2 = h_find_val(backward_cache, "db2");

    /**
     * UPDATE: w_i = w_i - lr  * D (gradients)
     */

    // apply: lr * D
    prsm_tensor_apply_scale_add(dw1, learning_rate, 0);
    prsm_tensor_apply_scale_add(db1, learning_rate, 0);
    prsm_tensor_apply_scale_add(dw2, learning_rate, 0);
    prsm_tensor_apply_scale_add(db2, learning_rate, 0);

    // printf("shape of  w1: (%zu, %zu)\n", prsm_tensor_shape(w1)[0], prsm_tensor_shape(w1)[1]);
    // printf("shape of dw1: (%zu, %zu)\n", prsm_tensor_shape(dw1)[0], prsm_tensor_shape(dw1)[1]);
    // printf("shape of  b1: (%zu, %zu)\n", prsm_tensor_shape(b1)[0], prsm_tensor_shape(b1)[1]);
    // printf("shape of db1: (%zu, %zu)\n", prsm_tensor_shape(db1)[0], prsm_tensor_shape(db1)[1]);
    // printf("shape of  w2: (%zu, %zu)\n", prsm_tensor_shape(w2)[0], prsm_tensor_shape(w2)[1]);
    // printf("shape of dw2: (%zu, %zu)\n", prsm_tensor_shape(dw2)[0], prsm_tensor_shape(dw2)[1]);
    // printf("shape of  b2: (%zu, %zu)\n", prsm_tensor_shape(b2)[0], prsm_tensor_shape(b2)[1]);
    // printf("shape of db2: (%zu, %zu)\n", prsm_tensor_shape(db2)[0], prsm_tensor_shape(db2)[1]);

    // transpose
    prsm_tensor_transpose(dw1);
    prsm_tensor_transpose(dw2);

    // w_i -= lr * D
    prsm_tensor_sub(w1, w1, dw1);
    prsm_tensor_sub(b1, b1, db1);
    prsm_tensor_sub(w2, w2, dw2);
    prsm_tensor_sub(b2, b2, db2);

    return params;
}

