#ifndef TIMECUBE_H_
#define TIMECUBE_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum timecube_cpu_type_e {
	TIMECUBE_CPU_NONE   = 0
#if defined(__i386) || defined(_M_IX86) || defined(_M_X64) || defined(__x86_64__)
	,TIMECUBE_CPU_SSE41  = 1,
	TIMECUBE_CPU_AVX2   = 2,
	TIMECUBE_CPU_AVX512 = 3
#endif
} timecube_cpu_type_e;

typedef enum timecube_pixel_type_e {
	TIMECUBE_PIXEL_BYTE,
	TIMECUBE_PIXEL_WORD,
	TIMECUBE_PIXEL_HALF,
	TIMECUBE_PIXEL_FLOAT
} timecube_pixel_type_e;

typedef enum timecube_pixel_range_e {
	TIMECUBE_RANGE_INTERNAL = -1,
	TIMECUBE_RANGE_LIMITED  = 0,
	TIMECUBE_RANGE_FULL     = 1
} timecube_pixel_range_e;

typedef enum timecube_lut_format_e {
	TIMECUBE_LUT_ADOBE_CUBE  /**< Adobe Cube LUT. */
} timecube_lut_format_e;


typedef struct timecube_lut timecube_lut;

timecube_lut *timecube_lut_read(const void *data, size_t size, timecube_lut_format_e format);

timecube_lut *timecube_lut_from_file(const char *path);

const char *timecube_lut_get_title(const timecube_lut *ptr);
int timecube_lut_set_title(timecube_lut *ptr, const char *title);

void timecube_lut_get_dimensions(const timecube_lut *ptr, size_t *dim, int *is_3d);
int timecube_lut_set_dimensions(timecube_lut *ptr, size_t dim, int is_3d);

void timecube_lut_get_domain(const timecube_lut *ptr, float min[3], float max[3]);
void timecube_lut_set_domain(timecube_lut *ptr, const float min[3], const float max[3]);

void timecube_lut_get_entry(const timecube_lut *ptr, unsigned r, unsigned g, unsigned b, float entry[3]);
void timecube_lut_set_entry(timecube_lut *ptr, unsigned r, unsigned g, unsigned b, const float entry[3]);

void timecube_lut_free(timecube_lut *ptr);


typedef struct timecube_filter timecube_filter;

typedef struct timecube_filter_params {
	timecube_pixel_type_e src_type;
	timecube_pixel_range_e src_range;
	unsigned src_depth;
	timecube_pixel_type_e dst_type;
	timecube_pixel_range_e dst_range;
	unsigned dst_depth;
} timecube_filter_params;

typedef struct timecube_filter_context {
	union {
		void *d;
		unsigned char buf[64];
	} u;
} timecube_filter_context;

timecube_filter *timecube_filter_create(const timecube_lut *lut, timecube_cpu_type_e cpu);

int timecube_filter_supports_type(const timecube_filter *ptr, timecube_pixel_type_e type);

int timecube_filter_create_context(const timecube_filter *ptr, const timecube_filter_params *params, timecube_filter_context *ctx);

void timecube_filter_context_apply(const timecube_filter_context *ctx, const void * const src[3], void * const dst[3], void *tmp, unsigned n);

void timecube_filter_free(timecube_filter *ptr);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TIMECUBE_H_ */
