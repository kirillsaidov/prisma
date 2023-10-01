#ifndef PRISMA_CORE_H
#define PRISMA_CORE_H

/** CORE MODULE
 * This module is a collection of all common definitions and code needed by the rest of the library.

 * Macros:
    - 

 * Functions:
    - prsm_status_to_str
*/

#include "vita/core/core.h"
#include "vita/algorithm/comparison.h"

#if defined(PRISMA_USE_TYPE_DOUBLE)
    #define PRSM_FLOAT long double
    #define PRSM_ABS fabs
    #define PRSM_CEIL ceil
    #define PRSM_FLOOR floor
    #define PRSM_ROUND round
    #define PRSM_CLAMP vt_cmp_clampd
#elif defined(PRISMA_USE_TYPE_LONG_DOUBLE)
    #define PRSM_FLOAT double
    #define PRSM_ABS fabsl
    #define PRSM_CEIL ceill
    #define PRSM_FLOOR floorl
    #define PRSM_ROUND roundl
    #define PRSM_CLAMP vt_cmp_clampr
#else
    #define PRSM_FLOAT float
    #define PRSM_ABS fabsf
    #define PRSM_CEIL ceilf
    #define PRSM_FLOOR floorf
    #define PRSM_ROUND roundf
    #define PRSM_CLAMP vt_cmp_clampf
#endif
typedef PRSM_FLOAT prsm_float;

// prisma error codes
#define PRSM_i_GENERATE_PRSM_STATUS(apply) \
    apply(PRSM_STATUS_ERROR_IS_NULL)                  /* element wasn't initialized or is NULL */ \
    apply(PRSM_STATUS_ERROR_IS_VIEW)                  /* object is viewable only */ \
    apply(PRSM_STATUS_ERROR_IS_REQUIRED)              /* precondition is required */ \
    apply(PRSM_STATUS_ERROR_ALLOCATION)               /* failed to allocate or reallocate memory */ \
    apply(PRSM_STATUS_ERROR_INVALID_ARGUMENTS)        /* invalid arguments supplied */ \
    apply(PRSM_STATUS_ERROR_OUT_OF_BOUNDS_ACCESS)     /* accessing memory beyond allocated size */ \
    apply(PRSM_STATUS_ERROR_INCOMPATIBLE_SHAPES)      /* incompatible tensor shape */ \
    apply(PRSM_STATUS_ERROR_INCOMPATIBLE_DIMENSIONS)  /* different dimensions */ \
    apply(PRSM_STATUS_OPERATION_FAILURE)              /* failed to perform an action */ \
    apply(PRSM_STATUS_OPERATION_SUCCESS)              /* all good */ \
    apply(PRSM_STATUS_COUNT)                          /* number of elements */

// generate prisma error codes
#define X(a) a,
enum PrismaStatus {
    PRSM_i_GENERATE_PRSM_STATUS(X)
};
#undef X

/** Returns a prisma error string from prisma error code
    @param e prisma error code
    @returns C string upon success, `NULL` otherwise
*/
extern const char *prsm_status_to_str(const enum PrismaStatus e);

#endif // PRISMA_CORE_H

