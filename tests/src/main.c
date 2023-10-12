#include <stdio.h>
#include <stdlib.h>
#include "vita/vita.h"
#include "prisma/prisma.h"

#define TEST(func) { printf("TESTING: %s\n", #func); func(); }

static vt_mallocator_t *alloctr = NULL;

void test_tensor(void);

int main(void) {
    vt_version_t 
        vt_v = vt_version_get(),
        prsm_v = prsm_version_get();
    printf("Vita (%s) | Prisma (%s)\n", vt_v.str, prsm_v.str);
    
    alloctr = vt_mallocator_create();
    {
        printf("----\n");

        TEST(test_tensor);

        printf("----\nDONE.\n");
    }
    vt_mallocator_destroy(alloctr);
    return 0;
}


/* ------ TESTS ------ */

void test_tensor(void) {
    prsm_tensor_t *v0 = prsm_tensor_create_vec(alloctr, 5);
    prsm_tensor_display(v0, NULL);

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

    // sub
    prsm_tensor_t *v3 = prsm_tensor_sub(NULL, v2, v1);
    assert(prsm_tensor_calc_sum(v3) == 7);

    // prsm_tensor_t *v4 = prsm_tensor_mul(NULL, v0, v1); // fails: use prsm_tensor_dot for vectors
    prsm_float dotprod = prsm_tensor_dot(v1, v2);
    assert(dotprod == 38);

    /*
     * MATRIX
     */

    // resize vector to matrix
    prsm_tensor_t *mat0 = prsm_tensor_dup(v3);
    prsm_tensor_resize(mat0, 2, 3, 3);
    prsm_tensor_set_identity(mat0);
    assert(prsm_tensor_calc_sum(mat0) == 3);

    // set/get
    prsm_tensor_set_val(mat0, vt_index_2d_to_1d(2, 0, mat0->shape[1]), 7);
    assert(prsm_tensor_get_val(mat0, vt_index_2d_to_1d(2, 0, mat0->shape[1])) == (float)7);
    prsm_tensor_display(mat0, NULL);

    // rand
    prsm_tensor_t *mat1 = prsm_tensor_create(alloctr, 2, 2, 3);
    prsm_tensor_rand(mat1);
    printf("RAND:\n"); prsm_tensor_display(mat1, NULL);
    prsm_tensor_rand_uniform(mat1, 0, 1);
    printf("RAND UNIFORM:\n"); prsm_tensor_display(mat1, NULL);
    VT_FOREACH(i, 0, prsm_tensor_size(mat1)) prsm_tensor_set_val(mat1, i, i);
    printf("INCREMENTAL:\n"); prsm_tensor_display(mat1, NULL);

    // multiplication
    prsm_tensor_t *mat2 = prsm_tensor_mul(NULL, mat1, mat0);
    printf("MAT1 *:\n"); prsm_tensor_display(mat1, NULL);
    printf("* MAT0:\n"); prsm_tensor_display(mat0, NULL);
    printf("= MAT3:\n"); prsm_tensor_display(mat2, NULL);
    assert(prsm_tensor_get_val(mat2, vt_index_2d_to_1d(0, 0, mat2->shape[1])) == 14);
    assert(prsm_tensor_get_val(mat2, vt_index_2d_to_1d(1, 0, mat2->shape[1])) == 38);

    // scale add
    prsm_tensor_apply_scale_add(mat2, 2, 1);
    prsm_tensor_display(mat2, NULL);
    assert(prsm_tensor_get_val(mat2, vt_index_2d_to_1d(0, 0, mat2->shape[1])) == 29);
    assert(prsm_tensor_get_val(mat2, vt_index_2d_to_1d(1, 0, mat2->shape[1])) == 77);
}




