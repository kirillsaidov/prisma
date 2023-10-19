#include "main.h"

prsm_tensor_t *perceptron_load_data(const char *const filename);
prsm_float threshold05(const prsm_float x);

void run_perceptron(vt_mallocator_t *alloctr) {
    vt_debug_redirect_output("debug.log");

    VT_LOG_INFO("Loading data.");
    prsm_tensor_t *data = perceptron_load_data("assets/perceptron_data_OR.txt");

    VT_LOG_INFO("Data loaded:");
    prsm_tensor_display(data, NULL);

    VT_LOG_INFO("Creating tensors...");
    prsm_tensor_t *weights, *x_train, *y_train;
    weights = prsm_tensor_create_vec(alloctr, prsm_tensor_shape(data)[1]);
    x_train = prsm_tensor_create_mat(alloctr, prsm_tensor_shape(data)[0], prsm_tensor_shape(data)[1]);
    y_train = prsm_tensor_create_vec(alloctr, prsm_tensor_shape(data)[0]);

    VT_LOG_INFO("Randomizing weights...");
    prsm_tensor_rand(weights);
    prsm_tensor_display(weights, NULL);

    VT_LOG_INFO("Initializing input data...");
    VT_FOREACH(i, 0, data->shape[0]) {
        VT_FOREACH(j, 0, data->shape[1]) { 
            prsm_tensor_set_val(
                x_train, 
                vt_index_2d_to_1d(i, j, x_train->shape[1]), 
                j == data->shape[1] - 1 ? 1 : prsm_tensor_get_val(data, vt_index_2d_to_1d(i, j, data->shape[1]))
            );
        }
    }
    prsm_tensor_display(x_train, NULL);

    VT_LOG_INFO("Initializing target data...");
    VT_FOREACH(i, 0, data->shape[0]) {
        y_train->data[i] = prsm_tensor_get_val(data, vt_index_2d_to_1d(i, data->shape[1]-1, data->shape[1]));
    }
    prsm_tensor_display(y_train, NULL);

    VT_LOG_INFO("Initializing model parameters...");
    const size_t epochs = 210;
    const prsm_float alpha = 0.05;

    VT_LOG_INFO("  alpha      = %.2f", alpha);
    VT_LOG_INFO("  activation = %s", VT_STRING_OF(prsm_activation_sigmoid));
    VT_LOG_INFO("  loss       = %s", VT_STRING_OF(prsm_cost_mse));
    VT_LOG_INFO("  epochs     = %zu", epochs);

    VT_LOG_INFO("Training model...");
    prsm_tensor_t *z = prsm_tensor_dup(y_train);
    prsm_tensor_t *yhat = prsm_tensor_dup(y_train);
    prsm_tensor_t *error = prsm_tensor_dup(y_train);
    prsm_tensor_t *costs = prsm_tensor_dup(y_train);
    VT_FOREACH(epoch, 0, epochs) {
        // zero out values
        prsm_tensor_set_zeros(z);
        prsm_tensor_set_zeros(yhat);
        prsm_tensor_set_zeros(error);

        // z = w*xT + b
        z = prsm_tensor_mul(z, x_train, weights);

        // yhat = activate(z)
        prsm_tensor_assign(yhat, z);
        prsm_tensor_apply_func(yhat, prsm_activation_relu);

        // calculate error: (y - yhat) * activation_derrivative(yhat)
        prsm_tensor_sub(error, y_train, yhat);
        VT_FOREACH(i, 0, prsm_tensor_size(error)) {
            error->data[i] *= prsm_activation_relu_d(yhat->data[i]);
        }
        
        // learn
        prsm_tensor_apply_scale_add(error, alpha, 0);
        VT_FOREACH(i, 0, prsm_tensor_size(error)) {
            VT_FOREACH(j, 0, prsm_tensor_size(weights)) {
                const prsm_float val = prsm_tensor_get_val(x_train, vt_index_2d_to_1d(i, j, x_train->shape[1]));
                weights->data[j] += alpha * error->data[i] * val;
            }
        }

        // calculate mse
        const prsm_float mse = prsm_cost_mse(yhat, y_train);

        // calculate accuracy
        prsm_float accuracy = 0;
        prsm_tensor_apply_func(yhat, threshold05);
        VT_FOREACH(i, 0, prsm_tensor_size(yhat)) {
            if (yhat->data[i] == y_train->data[i]) accuracy += 1;
        }
        accuracy /= prsm_tensor_size(yhat);
        VT_LOG_INFO("  Epoch %3zu | Error: %.2f | Accuracy: %.2f", epoch, mse, accuracy);
    }

    VT_LOG_INFO("Initializing test inputs...");
    prsm_tensor_t *x_test = prsm_tensor_create_mat(alloctr, 4, 3);
    prsm_tensor_assign_array(x_test, (prsm_float[]) { 
        0, 1, 1,
        1, 0, 1,
        0, 0, 1,
        1, 1, 1
    }, 12);
    prsm_tensor_display(x_test, NULL);

    VT_LOG_INFO("Running predictions...");
    yhat = prsm_tensor_mul(yhat, x_test, weights);
    prsm_tensor_apply_func(yhat, threshold05);
    prsm_tensor_display(yhat, NULL);

    VT_LOG_INFO("Done.");
}

prsm_tensor_t *perceptron_load_data(const char *const filename) {
    VT_LOG_ASSERT(vt_path_exists(filename), "File <%s> does not exist!");

    VT_LOG_INFO("Reading dataset: %s", filename);
    char raw_data[VT_STR_TMP_BUFFER_SIZE] = {0};
    vt_file_read_to_buffer(filename, raw_data, VT_STR_TMP_BUFFER_SIZE);

    // split by line
    vt_str_t tmp = vt_str_create_static(raw_data);
    vt_plist_t *list = vt_plist_create(VT_ARRAY_DEFAULT_INIT_ELEMENTS, alloctr);
    list = vt_str_split(list, &tmp, "\n");

    size_t rows, cols;
    sscanf(vt_str_z(vt_plist_get(list, 0)), " %zu x %zu ", &rows, &cols);
    VT_LOG_INFO("Dataset size (%zu, %zu)", rows, cols);

    // save data
    prsm_tensor_t *data = prsm_tensor_create_mat(alloctr, rows, cols);

    // load data to tensor
    VT_FOREACH(i, 1, vt_plist_len(list)) {
        prsm_tensor_t tview = prsm_tensor_make_view_vec(data, i-1);
        sscanf(vt_str_z(vt_plist_get(list, i)), " %f %f %f ", &tview.data[0], &tview.data[1], &tview.data[2]);
    }

    return data;
}

prsm_float threshold05(const prsm_float x) {
    return x <= 0.5 ? 0 : 1;
}

