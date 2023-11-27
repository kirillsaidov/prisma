#ifndef TEST_MAIN_H
#define TEST_MAIN_H

#include <stdio.h>
#include <stdlib.h>
#include "vita/vita.h"
#include "prisma/prisma.h"

#define CACHE_FOLDER "assets/cache/"

static vt_mallocator_t *alloctr = NULL;

// key-value pair
typedef struct {
    char key[256];
    void *value;
} h_keyval_t;

/// @brief Finds the value given key by iteration over a list
/// @param list plist_t
/// @param key c string
/// @return void*
void *h_find_val(vt_vec_t *list, char *key) {
    const size_t len = vt_vec_len(list);
    VT_FOREACH(i, 0, len) {
        h_keyval_t *keyval = vt_vec_get(list, i);
        if (vt_str_equals_z(key, keyval->key)) return keyval->value;
    }
    assert(0 && "h_find_val could not find key!");
}

#endif // TEST_MAIN_H

