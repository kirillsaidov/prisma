#include "prisma/core/version.h"

vt_version_t prsm_version_get(void) {
    vt_version_t v = {
        .major = PRSM_PRISMA_VERSION_MAJOR,
        .minor = PRSM_PRISMA_VERSION_MINOR,
        .patch = PRSM_PRISMA_VERSION_PATCH,
        .str = PRSM_PRISMA_VERSION
    };

    return v;
}

