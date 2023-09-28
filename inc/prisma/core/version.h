#ifndef PRISMA_CORE_VERSION_H
#define PRISMA_CORE_VERSION_H

/** VERSION MODULE
    - prsm_version_get
*/

#include "vita/core/version.h"

// defines
#define PRSM_PRISMA_VERSION_MAJOR 0
#define PRSM_PRISMA_VERSION_MINOR 1
#define PRSM_PRISMA_VERSION_PATCH 0
#define PRSM_PRISMA_VERSION VT_STRING_OF(VT_PCAT(VT_PCAT(VT_PCAT(VT_PCAT(PRSM_PRISMA_VERSION_MAJOR, .), PRSM_PRISMA_VERSION_MINOR), .), PRSM_PRISMA_VERSION_PATCH))

/** Query Prisma version
    @returns vt_version_t struct containing major, minor, patch and full version str
*/
extern vt_version_t prsm_version_get(void);

#endif // PRISMA_CORE_VERSION_H

