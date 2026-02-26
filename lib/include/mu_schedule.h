#ifndef __MU_SCHEDULE_H__
#define __MU_SCHEDULE_H__

#include <stdint.h>

extern "C" {

typedef void (*mu_schedule_callback)(void *arg,
                                     uint32_t tid_in_threadblock,
                                     uint32_t threads_per_threadblock,
                                     uint32_t threadblock_id);

void mu_schedule(mu_schedule_callback entry_point, void * arg);

} // extern "C"

#endif // __MU_SCHEDULE_H__
