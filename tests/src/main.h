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
} dict_keyval_t;

/// @brief Finds the value given key by iterating over a list
/// @param list vt_vec_t
/// @param key string key
/// @return void* or NULL if element not found
void *dict_find_val(vt_vec_t *list, char *key) {
    const size_t len = vt_vec_len(list);
    VT_FOREACH(i, 0, len) {
        dict_keyval_t *keyval = vt_vec_get(list, i);
        if (vt_str_equals_z(key, keyval->key)) return keyval->value;
    }
    return NULL;
}

/// @brief Updates the value given key by iterating over a list
/// @param list vt_vec_t
/// @param key string key
/// @param value value
/// @return None
void dict_update_val(vt_vec_t *list, char *key, void *value) {
    // find value and update; return;
    const size_t len = vt_vec_len(list);
    VT_FOREACH(i, 0, len) {
        dict_keyval_t *keyval = vt_vec_get(list, i);
        if (vt_str_equals_z(key, keyval->key)) {
            keyval->value = value;
            return;
        }
    }
    
    // add to dict
    dict_keyval_t item = {.value = value};
    strncpy(item.key, key, sizeof(item.key));
    vt_vec_push(list, &item);
}

#endif // TEST_MAIN_H

