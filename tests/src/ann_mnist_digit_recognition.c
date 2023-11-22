#include "main.h"
#include <curl/curl.h>

#define MNIST_TRAIN "mnist_train.csv"
#define MNIST_TEST "mnist_test.csv"

void ann_download_csv(const char *const url, const char *const filepath);
void ann_read_mnist_data(const char *const file, prsm_tensor_t *x, prsm_tensor_t *y);

void run_ann_mnist_digit_recognition(void) {
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
    const size_t num_rows = 10;
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

    VT_LOG_INFO("Creating weights and biases...");
    const size_t layer_output_size = 10;
    const size_t layer_hidden_size = 100;
    prsm_tensor_t *w1, *w2, *b1, *b2;
    w1 = prsm_tensor_create_mat(alloctr, layer_hidden_size, num_features);
    w2 = prsm_tensor_create_mat(alloctr, layer_output_size, layer_hidden_size);
    b1 = prsm_tensor_create_vec(alloctr, layer_hidden_size);
    b2 = prsm_tensor_create_vec(alloctr, layer_output_size);

    VT_LOG_INFO("NN created: (768, 100, 10)");
    VT_LOG_INFO("Randomizing weights...");
    prsm_tensor_rand(w1);
    prsm_tensor_rand(w2);
    prsm_tensor_set_all(b1, 1);
    prsm_tensor_set_all(b2, 1);

    VT_LOG_INFO("Initializing model parameters...");
    const size_t epochs = 1;
    const prsm_float alpha = 0.05;

    VT_LOG_INFO("  alpha         = %.2f", alpha);
    VT_LOG_INFO("  activation l1 = %s", VT_STRING_OF(prsm_math_sigmoid));
    VT_LOG_INFO("  activation l2 = %s", VT_STRING_OF(prsm_activate_softmax));
    VT_LOG_INFO("  loss          = %s", VT_STRING_OF(prsm_loss_cce));
    VT_LOG_INFO("  epochs        = %zu", epochs);

    VT_LOG_INFO("Start training...");
    // prsm_tensor_t *z1 = prsm_tensor_create_vec(alloctr, 10);
    // prsm_tensor_t *z2 = prsm_tensor_create_vec(alloctr, 10);
    // prsm_tensor_t *a1 = prsm_tensor_dup(z);
    // prsm_tensor_t *a2 = prsm_tensor_dup(z);
    // prsm_tensor_t *error = prsm_tensor_dup(z);
    // prsm_tensor_t *delta = prsm_tensor_dup(z);
    prsm_tensor_transpose(w1);
    prsm_tensor_transpose(w2);
    VT_FOREACH(epoch, 0, epochs) {
        /* -----------------------
        * FORWARD
        */

        // z1 = w*xT + b
        prsm_tensor_t *a1 = prsm_tensor_mul(NULL, x_train, w1);

        // add bias
        VT_FOREACH(i, 0, prsm_tensor_shape(a1)[0]) {
            prsm_tensor_t t = prsm_tensor_make_view_vec(a1, i);
            VT_FOREACH(j, 0, prsm_tensor_size(&t)) t.data[j] += b1->data[j];
        }

        // activate
        prsm_tensor_apply_func(a1, prsm_math_relu);

        // z2 = w*xT + b
        prsm_tensor_t *z2 = prsm_tensor_mul(NULL, a1, w2);

        // add bias
        VT_FOREACH(i, 0, prsm_tensor_shape(a1)[0]) {
            prsm_tensor_t t = prsm_tensor_make_view_vec(a1, i);
            VT_FOREACH(j, 0, prsm_tensor_size(&t)) t.data[j] += b1->data[j];
        }

        // activate
        prsm_tensor_t *a2 = prsm_tensor_create_mat(alloctr, 10, 10);
        VT_FOREACH(i, 0, prsm_tensor_shape(a2)[0]) {
            prsm_tensor_t a2_i = prsm_tensor_make_view_vec(a2, i);
            prsm_tensor_t z2_i = prsm_tensor_make_view_vec(z2, i);
            prsm_activate_softmax(&a2_i, &z2_i);
        }

        prsm_tensor_display(a2, NULL);

        prsm_tensor_destroy(a1);
        prsm_tensor_destroy(z2);
        prsm_tensor_destroy(a2);
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

void ann_plist_str_destroy(vt_plist_t *p);
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
                const vt_str_t *item = (vt_str_t*)vt_plist_get(line_items, i);

                // convert to prsm_float
                const prsm_float value = vt_conv_str_to_f(vt_str_z(item));
                
                // save values
                if (i == 0) prsm_tensor_set_val(y, vt_index_2d_to_1d(current_row, value, 10), 1);
                else prsm_tensor_set_val(x, vt_index_2d_to_1d(current_row, i-1, read_cols), value);
            }
            // ----------------

            // ---------- note ----------
            // const char *ptr = buffer;
            // prsm_float value = 0;
            // sscanf(ptr, "%f ", &value);
            // printf("%.0f ", value);
            // while((ptr = strstr(ptr, ","))) {
            //     ptr += 1;
            //     sscanf(ptr, "%f ", &value);
            //     printf("%.0f ", value);
            // }
            // printf("\n");
            // ---------- note ----------

            // update
            current_row++;
        }
    } 
    fclose(fp);
}

