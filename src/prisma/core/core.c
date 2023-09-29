#include "prisma/core/core.h"

// generate prisma error strings
#define X(a) VT_STRING_OF(a),
static const char *const prsm_error_str[] = {
    PRSM_i_GENERATE_PRSM_STATUS(X)
};
#undef X

const char *prsm_status_to_str(const enum PrismaStatus e) {
    if (e < PRSM_STATUS_COUNT) {
        return prsm_error_str[e];
    }

    return NULL;
}

