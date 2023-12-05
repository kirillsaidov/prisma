#include "main.h"
#include <curl/curl.h>

#define MNIST_TRAIN "mnist_train.csv"
#define MNIST_TEST "mnist_test.csv"

void ann_download_csv(const char *const url, const char *const filepath);
void ann_read_mnist_data(const char *const file, prsm_tensor_t *x, prsm_tensor_t *y);
vt_vec_t *ann_model_init_params(const size_t N, const size_t n_x, const size_t n_h, const size_t n_y);
vt_vec_t *ann_model_forward(vt_vec_t *params, prsm_tensor_t *x);
vt_vec_t *ann_model_backward(prsm_tensor_t *x, prsm_tensor_t *y, vt_vec_t *params);
vt_vec_t *ann_model_update(vt_vec_t *params, vt_vec_t *backward_cache, const prsm_float learning_rate);
prsm_float ann_cost(const prsm_tensor_t *const pred, const prsm_tensor_t *const target);
prsm_float ann_accuracy(prsm_tensor_t *const pred, const prsm_tensor_t *const target, const bool round);

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
    const size_t num_rows = 8;
    const size_t num_features = 28*28;
    const size_t output_size = 10;
    prsm_tensor_t *x_train, *y_train, *x_test, *y_test;
    x_train = prsm_tensor_create_mat(alloctr, num_rows, num_features);
    y_train = prsm_tensor_create_mat(alloctr, num_rows, output_size);
    x_test = prsm_tensor_create_mat(alloctr, (size_t)(num_rows/2), num_features);
    y_test = prsm_tensor_create_mat(alloctr, (size_t)(num_rows/2), output_size);

    VT_LOG_INFO("Loading data into memory: %zu instances", num_rows);
    ann_read_mnist_data(CACHE_FOLDER MNIST_TRAIN, x_train, y_train);
    ann_read_mnist_data(CACHE_FOLDER MNIST_TEST, x_test, y_test);

    VT_LOG_INFO("Normalize the data...");
    prsm_tensor_apply_scale_add(x_train, 1.0/255.0, 0);
    prsm_tensor_apply_scale_add(x_test, 1.0/255.0, 0);

    VT_LOG_INFO("Initializing parameters...");
    const size_t layer_hidden_size = 100;
    vt_vec_t *params = ann_model_init_params(num_rows, num_features, layer_hidden_size, output_size);

    VT_LOG_INFO("Initializing model options...");
    const size_t epochs = 1;
    const prsm_float alpha = 0.05;

    VT_LOG_INFO("\talpha         = %.2f", alpha);
    VT_LOG_INFO("\tactivation l2 = %s", VT_STRING_OF(prsm_math_relu));
    VT_LOG_INFO("\tactivation l3 = %s", VT_STRING_OF(prsm_activate_ssoftmax));
    VT_LOG_INFO("\tloss          = %s", VT_STRING_OF(prsm_loss_cce));
    VT_LOG_INFO("\tepochs        = %zu", epochs);

    VT_LOG_INFO("Start training...");
    VT_FOREACH(epoch, 0, epochs) {
        /* -----------------------
        * FORWARD
        */
        params = ann_model_forward(params, x_train);

        /* -----------------------
        * COST AND ACCURACY
        */
        prsm_tensor_t *yhat = dict_find_val(params, "a2");
        const prsm_float cost = ann_cost(yhat, y_train);
        const prsm_float accuracy = ann_accuracy(yhat, y_train, true);
        VT_LOG_INFO("\tEpoch %3zu | Error: %.2f | Accuracy: %.2f", epoch, cost, accuracy);

        /* -----------------------
        * BACKWARD
        */
        params = ann_model_backward(x_train, y_train, params);

        /* -----------------------
        * UPDATE
        */
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

vt_vec_t *ann_model_init_params(const size_t N, const size_t n_x, const size_t n_h, const size_t n_y) {
    VT_LOG_INFO("\tCreating weights and biases...");
    prsm_tensor_t *w1, *w2, *b1, *b2;
    w1 = prsm_tensor_create_mat(alloctr, n_x, n_h);
    b1 = prsm_tensor_create_vec(alloctr, n_h);
    w2 = prsm_tensor_create_mat(alloctr, n_h, n_y);
    b2 = prsm_tensor_create_vec(alloctr, n_y);

    VT_LOG_INFO("\tModel created: (L1: %zu, L2: %zu, L3: %zu)", n_x, n_h, n_y);

    VT_LOG_INFO("\tRandomizing weights...");
    prsm_tensor_rand(w1);
    prsm_tensor_rand(w2);
    prsm_tensor_set_ones(b1);
    prsm_tensor_set_ones(b2);

    VT_LOG_INFO("\tFinalizing...");
    prsm_tensor_t *z1, *a1, *z2, *a2;
    z1 = prsm_tensor_create_mat(alloctr, N, n_h);
    a1 = prsm_tensor_create_mat(alloctr, N, n_h);
    z2 = prsm_tensor_create_mat(alloctr, N, n_y);
    a2 = prsm_tensor_create_mat(alloctr, N, n_y);

    // add to dict
    vt_vec_t *params = vt_vec_create(10, sizeof(dict_keyval_t), alloctr);
    dict_update_val(params, "w1", w1);
    dict_update_val(params, "b1", b1);
    dict_update_val(params, "w2", w2);
    dict_update_val(params, "b2", b2);
    dict_update_val(params, "z1", z1);
    dict_update_val(params, "a1", a1);
    dict_update_val(params, "z2", z2);
    dict_update_val(params, "a2", a2);

    return params;
}

vt_vec_t *ann_model_forward(vt_vec_t *params, prsm_tensor_t *x) {
    // get params
    prsm_tensor_t *w1, *w2, *b1, *b2, *z1, *a1, *z2, *a2;
    w1 = dict_find_val(params, "w1");
    b1 = dict_find_val(params, "b1");
    w2 = dict_find_val(params, "w2");
    b2 = dict_find_val(params, "b2");
    z1 = dict_find_val(params, "z1");
    a1 = dict_find_val(params, "a1");
    z2 = dict_find_val(params, "z2");
    a2 = dict_find_val(params, "a2");

    /**
     * LAYER 2
     * 
     * note: layer 1 is the inputs itself
     */

    // z1 = x * w1 + b1
    z1 = prsm_tensor_dot(z1, x, w1);

    // add bias
    VT_FOREACH(i, 0, prsm_tensor_shape(z1)[0]) {
        prsm_tensor_t tview = prsm_tensor_make_view_vec(z1, i);
        prsm_tensor_add(&tview, &tview, b1);
    }

    // activate: a1 = relu(z1)
    a1 = prsm_activate_relu(a1, z1);

    /**
     * LAYER 3
     */

    // z2 = a1 * w2 + b2
    z2 = prsm_tensor_dot(z2, a1, w2);

    // add bias
    VT_FOREACH(i, 0, prsm_tensor_shape(z2)[0]) {
        prsm_tensor_t tview = prsm_tensor_make_view_vec(z2, i);
        prsm_tensor_add(&tview, &tview, b2);
    }

    // activate: a2 = softmax(z2)
    prsm_tensor_assign(a2, z2);
    VT_FOREACH(i, 0, prsm_tensor_shape(a2)[0]) {
        prsm_tensor_t tview = prsm_tensor_make_view_vec(a2, i);
        prsm_activate_ssoftmax(&tview, &tview);
    }
    // prsm_tensor_display(z2, NULL); // TODO: remove

    return params;
}

vt_vec_t *ann_model_backward(prsm_tensor_t *x, prsm_tensor_t *y, vt_vec_t *params) {
    // get params
    prsm_tensor_t *w1, *w2, *b1, *b2, *a1, *a2, *z1, *z2;
    w1 = dict_find_val(params, "w1");
    b1 = dict_find_val(params, "b1");
    w2 = dict_find_val(params, "w2");
    b2 = dict_find_val(params, "b2");
    a1 = dict_find_val(params, "a1");
    a2 = dict_find_val(params, "a2");
    z1 = dict_find_val(params, "z1");
    z2 = dict_find_val(params, "z2");

    // number of observations
    const float N = prsm_tensor_shape(x)[0];

    /**
     * LAYER 3
     */
    
    // DC = dc/da2 | (N, 10)
    prsm_tensor_t *DC = dict_find_val(params, "DC");
    if (!DC) {
        DC = prsm_tensor_dup(a2);
        dict_update_val(params, "DC", DC);
    }
    VT_FOREACH(i, 0, N) { // for each example
        prsm_tensor_t y_view = prsm_tensor_make_view_vec(y, i);
        prsm_tensor_t a2_view = prsm_tensor_make_view_vec(a2, i);
        prsm_tensor_t DC_view = prsm_tensor_make_view_vec(DC, i);
        prsm_loss_cce_d(&DC_view, &a2_view, &y_view);
    }

    printf("==> DC:\n");
    prsm_tensor_display(DC, NULL);

    // DA2 = da2/dz2 = ssoftmax'(z2) | (N, 10)
    prsm_tensor_t *DA2 = dict_find_val(params, "DA2");
    if (!DA2) {
        DA2 = prsm_tensor_dup(z2);
        dict_update_val(params, "DA2", DA2);
    }
    VT_FOREACH(i, 0, N) { // for each example
        prsm_tensor_t z2_view = prsm_tensor_make_view_vec(z2, i);
        prsm_tensor_t DA2_view = prsm_tensor_make_view_vec(DA2, i);
        prsm_activate_ssoftmax_d(&DA2_view, &z2_view);
    }

    printf("==> DA2:\n");
    prsm_tensor_display(DA2, NULL);
    
    // DZ2_dw2 = dz2/dw2 = a1 | (N, 100)
    prsm_tensor_t *DZ2_dw2 = a1;

    printf("==> DZ2_dw2:\n");
    prsm_tensor_display(DZ2_dw2, NULL);

    // DW2 = (DC * DA2_T * DZ2_dw2)_T * a2 | (N, 100)
    prsm_tensor_t *DW2 = dict_find_val(params, "DW2");
    if (!DW2) {
        DW2 = prsm_tensor_dup(w2);
        dict_update_val(params, "DW2", DW2);
    }
    
    prsm_tensor_transpose(DA2);
    {   
        // DC * DA2_T * DZ2
        prsm_tensor_t *tmp = prsm_tensor_dot(NULL, DC, DA2);    // (N, N)
        prsm_tensor_t *tmp2 = prsm_tensor_dot(NULL, tmp, DZ2_dw2);  // (N, 100)

        // (DC * DA2_T * DZ2)_T * a2
        prsm_tensor_transpose(tmp2);
        DW2 = prsm_tensor_dot(DW2, tmp2, a2);                   // (100, 10)

        prsm_tensor_destroy(tmp);
        prsm_tensor_destroy(tmp2);
    }
    prsm_tensor_transpose(DA2);

    printf("==> DW2:\n");
    prsm_tensor_display(DW2, NULL);

    /**
     * LAYER 2
     */

    // DZ2_da1 = dz2/da1 = w2 | (100, 10)
    prsm_tensor_t *DZ2_da1 = w2;

    printf("==> DZ2_da1:\n");
    prsm_tensor_display(DZ2_da1, NULL);

    // DA1 = da1/dz1 = relu'(z1) | (N, 100)
    prsm_tensor_t *DA1 = dict_find_val(params, "DA1");
    if (!DA1) {
        DA1 = prsm_tensor_dup(z1);
        dict_update_val(params, "DA1", DA1);
    }
    VT_FOREACH(i, 0, N) { // for each example
        prsm_tensor_t z1_view = prsm_tensor_make_view_vec(z1, i);
        prsm_tensor_t DA1_view = prsm_tensor_make_view_vec(DA1, i);
        prsm_activate_ssoftmax_d(&DA1_view, &z1_view);
    }

    printf("==> DA1:\n");
    prsm_tensor_display(DA1, NULL);

    // DZ1 = dz1/dw1 = x
    prsm_tensor_t *DZ1 = x;

    printf("==> DZ1:\n");
    prsm_tensor_display(DZ1, NULL);

    // DW1 = DC * DA2 * DZ2_da1 * DA1 * DZ1 | (784, 100)

    return params;
}

vt_vec_t *ann_model_update(vt_vec_t *params, vt_vec_t *backward_cache, const prsm_float learning_rate) {
    // get params
    prsm_tensor_t *w1, *w2, *b1, *b2;
    w1 = dict_find_val(params, "w1");
    b1 = dict_find_val(params, "b1");
    w2 = dict_find_val(params, "w2");
    b2 = dict_find_val(params, "b2");

    // get backward_cache (gradients)
    prsm_tensor_t *dw1, *dw2, *db1, *db2;
    dw1 = dict_find_val(backward_cache, "dw1");
    db1 = dict_find_val(backward_cache, "db1");
    dw2 = dict_find_val(backward_cache, "dw2");
    db2 = dict_find_val(backward_cache, "db2");

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

prsm_float ann_cost(const prsm_tensor_t *const pred, const prsm_tensor_t *const target) {
    VT_ENFORCE(prsm_tensor_shapes_match(pred, target), "Shapes don't match!");

    // calculate overall cost
    prsm_float cost = 0;
    const size_t N = prsm_tensor_shape(target)[0];
    VT_FOREACH(i, 0, N) {
        prsm_tensor_t pred_view = prsm_tensor_make_view_vec(pred, i);
        prsm_tensor_t target_view = prsm_tensor_make_view_vec(target, i);
        cost += prsm_loss_cce(&pred_view, &target_view);
    }
    
    return cost/((prsm_float)N);
}

prsm_float ann_step_func(const prsm_float v) { return v > 0.5 ? 1 : 0; }
prsm_float ann_accuracy(prsm_tensor_t *const pred, const prsm_tensor_t *const target, const bool round) {
    VT_ENFORCE(prsm_tensor_shapes_match(pred, target), "Shapes don't match!");

    // round values
    if (round) prsm_tensor_apply_func(pred, ann_step_func);

    // calculate accuracy
    prsm_float accuracy = 0;
    const size_t N = prsm_tensor_shape(pred)[0];
    VT_FOREACH(i, 0, N) {
        const prsm_tensor_t pred_item = prsm_tensor_make_view_vec(pred, i);
        const prsm_tensor_t target_item = prsm_tensor_make_view_vec(target, i);
        if (prsm_tensor_equals(&pred_item, &target_item)) accuracy++;
    }

    return accuracy/((prsm_float)N);
}

