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

    prsm_tensor_t *v1 = prsm_tensor_dup(v0);
    assert(prsm_tensor_equals(v0, v1));

    prsm_tensor_t *v2 = prsm_tensor_add(v0, v1);
    assert(prsm_tensor_calc_sum(v2) == 14);
}




