#ifndef PRISMA_CORE_TENSOR_H
#define PRISMA_CORE_TENSOR_H

/** TENSOR MODULE
    - prsm_tensor_create    
    - prsm_tensor_destroy    
*/

#include "core.h"
#include "vita/allocator/mallocator.h"

typedef struct PrismaTensor {
    prsm_float *data;
    size_t length; 
} prsm_tensor_t;

extern prsm_tensor_t *prsm_tensor_create(const size_t length, struct VitaBaseAllocatorType *const alloctr);
extern void prsm_tensor_destroy(prsm_tensor_t *tensor);

#endif // PRISMA_CORE_TENSOR_H

