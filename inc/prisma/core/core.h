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
#include "vita/util/debug.h"
#include "vita/algorithm/comparison.h"

#if defined(PRISMA_USE_TYPE_DOUBLE)
    #define PRSM_FLOAT double
    #define PRSM_ABS fabs
    #define PRSM_CEIL ceil
    #define PRSM_FLOOR floor
    #define PRSM_ROUND round
    #define PRSM_POW pow
    #define PRSM_SQRT sqrt
    #define PRSM_CLAMP vt_cmp_clampd
    #define PRSM_MAX vt_cmp_maxd
    #define PRSM_MIN vt_cmp_mind
    #define PRSM_EXP exp
    #define PRSM_TANH tanh
    #define PRSM_LOG log
    #define PRSM_CONST_EPSILON __DBL_EPSILON__
#elif defined(PRISMA_USE_TYPE_LONG_DOUBLE)
    #define PRSM_FLOAT long double
    #define PRSM_ABS fabsl
    #define PRSM_CEIL ceill
    #define PRSM_FLOOR floorl
    #define PRSM_ROUND roundl
    #define PRSM_POW powl
    #define PRSM_SQRT sqrtl
    #define PRSM_CLAMP vt_cmp_clampr
    #define PRSM_MAX vt_cmp_maxr
    #define PRSM_MIN vt_cmp_minr
    #define PRSM_EXP expl
    #define PRSM_TANH tanhl
    #define PRSM_LOG logl
    #define PRSM_CONST_EPSILON __LDBL_EPSILON__
#else
    #define PRSM_FLOAT float
    #define PRSM_ABS fabsf
    #define PRSM_CEIL ceilf
    #define PRSM_FLOOR floorf
    #define PRSM_ROUND roundf
    #define PRSM_POW powf
    #define PRSM_SQRT sqrtf
    #define PRSM_CLAMP vt_cmp_clampf
    #define PRSM_MAX vt_cmp_maxf
    #define PRSM_MIN vt_cmp_minf
    #define PRSM_EXP expf
    #define PRSM_TANH tanhf
    #define PRSM_LOG logf
    #define PRSM_CONST_EPSILON __FLT_EPSILON__
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

