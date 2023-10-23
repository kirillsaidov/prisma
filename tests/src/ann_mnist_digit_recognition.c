#include "main.h"
#include <curl/curl.h>

#define MNIST_TRAIN "mnist_train.csv"
#define MNIST_TEST "mnist_test.csv"

void ann_download_csv(const char *const url, const char *const filepath);
void ann_read_mnist_data(prsm_tensor_t *x_train, prsm_tensor_t *y_train, prsm_tensor_t *x_test, prsm_tensor_t *y_test);

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
    const size_t load_rows = 1;
    prsm_tensor_t *x_train, *y_train, *x_test, *y_test;
    x_train = prsm_tensor_create_mat(alloctr, load_rows, 28*28 + 1); // +1 is for bias
    x_test = prsm_tensor_create_mat(alloctr, load_rows, 28*28 + 1);  // +1 is for bias
    y_train = prsm_tensor_create_vec(alloctr, load_rows);
    y_test = prsm_tensor_create_vec(alloctr, load_rows);

    VT_LOG_INFO("Loading data into memory...");
    ann_read_mnist_data(x_train, y_train, x_test, y_test);
    prsm_tensor_display(x_train, NULL);
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

const char* ann_get_field(char* line, int num);
void ann_read_mnist_data(prsm_tensor_t *x_train, prsm_tensor_t *y_train, prsm_tensor_t *x_test, prsm_tensor_t *y_test) {
    char buffer[2048];
    size_t current_row = 0;
    const size_t read_rows = prsm_tensor_shape(x_train)[0];
    const size_t read_cols = prsm_tensor_shape(x_train)[1];
    vt_str_t *line = vt_str_create_len(sizeof(buffer), alloctr);

    /**
     * READING TRAIN DATA
     */
    VT_LOG_INFO("Reading %s", MNIST_TRAIN);
    FILE *fp = fopen(CACHE_FOLDER MNIST_TRAIN, "r"); 
    {
        while(current_row < read_rows && fgets(buffer, sizeof(buffer), fp)) {
            current_row++;

            // split
            size_t current_col = 0;
            const char *entry;
            vt_str_set(line, buffer);
            while((entry = vt_str_find(line, ","))) {
                prsm_float val = -1;
                sscanf(entry+1, " %f,", &val);
                prsm_tensor_set_val(x_train, current_col++, val);
                // printf("%s\n", entry+1);
            }
        }
    } 
    fclose(fp);
}

const char* ann_get_field(char *line, int num) {
    const char* tok;
    for (tok = strtok(line, ","); tok && *tok; tok = strtok(NULL, ",\n")) {
        if (!--num) return tok;
    }
    return NULL;
}

