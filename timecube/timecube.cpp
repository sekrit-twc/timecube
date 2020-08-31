#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>
#include "cube.h"
#include "lut.h"
#include "timecube.h"

#ifdef _WIN32
  #include <Windows.h>
#endif

namespace {

struct FileCloser {
	void operator()(std::FILE *file) const { std::fclose(file); }
};

struct LutParams {
	const timecube::Lut *lut;
	timecube::PixelFormat src_fmt;
	timecube::PixelFormat dst_fmt;
};
static_assert(sizeof(LutParams) < sizeof(timecube_filter_context), "context too small");

#ifdef _WIN32
std::wstring utf8_to_utf16(const std::string &s)
{
	if (s.size() > static_cast<size_t>(INT_MAX))
		throw std::length_error{ "" };

	std::wstring us(s.size(), L'\0');

	int len = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()), &us[0], static_cast<int>(us.size()));
	if (!len)
		throw std::runtime_error{ "bad UTF-8/UTF-16 conversion" };

	us.resize(len);
	return us;
}
#endif

} // namespace


timecube_lut *timecube_lut_read(const void *data, size_t size, timecube_lut_format_e format)
{
	if (format != TIMECUBE_LUT_ADOBE_CUBE)
		return nullptr;

	return nullptr;
}

timecube_lut *timecube_lut_from_file(const char *path) try
{
	auto cube = std::make_unique<timecube::Cube>(timecube::read_cube_from_file(path));
	return cube.release();
} catch (...) {
	return nullptr;
}

const char *timecube_lut_get_title(const timecube_lut *ptr)
{
	return static_cast<const timecube::Cube *>(ptr)->title.c_str();
}

int timecube_lut_set_title(timecube_lut *ptr, const char *title) try
{
	static_cast<timecube::Cube *>(ptr)->title = title;
	return 0;
} catch (...) {
	return 1;
}

void timecube_lut_get_dimensions(const timecube_lut *ptr, size_t *dim, int *is_3d)
{
	const timecube::Cube *cube = static_cast<const timecube::Cube *>(ptr);
	*dim = cube->n;
	*is_3d = cube->is_3d;
}

int timecube_lut_set_dimensions(timecube_lut *ptr, size_t dim, int is_3d) try
{
	if (!is_3d && (dim < 2 || dim > 65536))
		return 1;
	if (is_3d && dim > 256)
		return 1;

	timecube::Cube *cube = static_cast<timecube::Cube *>(ptr);
	cube->n = static_cast<uint_least32_t>(dim);
	cube->is_3d = is_3d;
	cube->lut.clear();
	cube->lut.resize(is_3d ? dim * dim * dim : dim);
	return 0;
} catch (...) {
	return 1;
}

void timecube_lut_get_domain(const timecube_lut *ptr, float min[3], float max[3])
{
	const timecube::Cube *cube = static_cast<const timecube::Cube *>(ptr);
	std::copy_n(cube->domain_min, 3, min);
	std::copy_n(cube->domain_max, 3, max);
}

void timecube_lut_set_domain(timecube_lut *ptr, const float min[3], const float max[3])
{
	timecube::Cube *cube = static_cast<timecube::Cube *>(ptr);
	std::copy_n(min, 3, cube->domain_min);
	std::copy_n(max, 3, cube->domain_max);
}

void timecube_lut_get_entry(const timecube_lut *ptr, unsigned r, unsigned g, unsigned b, float entry[3])
{
	const timecube::Cube *cube = static_cast<const timecube::Cube *>(ptr);
	size_t idx = cube->is_3d ? b * cube->n * cube->n + g * cube->n + r : r * 3;
	std::copy_n(cube->lut.begin() + idx, 3, entry);
}

void timecube_lut_set_entry(timecube_lut *ptr, unsigned r, unsigned g, unsigned b, const float entry[3])
{
	timecube::Cube *cube = static_cast<timecube::Cube *>(ptr);
	size_t idx = cube->is_3d ? b * cube->n * cube->n + g * cube->n + r : r * 3;
	std::copy_n(entry, 3, cube->lut.begin() + idx);
}

void timecube_lut_free(timecube_lut *ptr)
{
	delete static_cast<timecube::Cube *>(ptr);
}

timecube_filter *timecube_filter_create(const timecube_lut *lut, timecube_cpu_type_e cpu) try
{
	const timecube::Cube *cube = static_cast<const timecube::Cube *>(lut);
	std::unique_ptr<timecube::Lut> ptr = timecube::create_lut_impl(*cube, cpu);
	return ptr.release();
} catch (...) {
	return nullptr;
}

int timecube_filter_supports_type(const timecube_filter *ptr, timecube_pixel_type_e type)
{
	const timecube::Lut *lut = static_cast<const timecube::Lut *>(ptr);
	return type == TIMECUBE_PIXEL_HALF ? lut->supports_half() : 1;
}

int timecube_filter_create_context(const timecube_filter *ptr, const timecube_filter_params *params, timecube_filter_context *ctx)
{
	if (params->src_type == TIMECUBE_PIXEL_BYTE && params->src_depth > 8)
		return 1;
	if (params->src_type == TIMECUBE_PIXEL_WORD && params->src_depth > 16)
		return 1;
	if (params->dst_type == TIMECUBE_PIXEL_BYTE && params->dst_depth > 8)
		return 1;
	if (params->dst_type == TIMECUBE_PIXEL_WORD && params->dst_depth > 16)
		return 1;
	if (params->src_type < 0 || params->dst_type < 0)
		return 1;
	if (params->src_type > TIMECUBE_PIXEL_FLOAT || params->dst_type > TIMECUBE_PIXEL_FLOAT)
		return 1;

	const timecube::Lut *lut = static_cast<const timecube::Lut *>(ptr);
	LutParams *p = new (ctx->u.buf) LutParams{};
	p->lut = lut;
	p->src_fmt.type = static_cast<timecube::PixelType>(params->src_type);
	p->src_fmt.fullrange = params->src_range == TIMECUBE_RANGE_FULL;
	p->src_fmt.depth = params->src_depth;
	p->dst_fmt.type = static_cast<timecube::PixelType>(params->dst_type);
	p->dst_fmt.fullrange = params->dst_range == TIMECUBE_RANGE_FULL;
	p->dst_fmt.depth = params->dst_depth;
	return 0;
}

void timecube_filter_context_apply(const timecube_filter_context *ctx, const void * const src[3], void * const dst[3], void *tmp, unsigned n) try
{
	const LutParams *p = reinterpret_cast<const LutParams *>(ctx->u.buf);
	const timecube::Lut *lut = p->lut;
	unsigned n_aligned = (n + 15) & ~15;
	float *src_f[3];
	float *dst_f[3];

	if (p->src_fmt.type == timecube::PixelType::FLOAT) {
		src_f[0] = const_cast<float *>(static_cast<const float *>(src[0]));
		src_f[1] = const_cast<float *>(static_cast<const float *>(src[1]));
		src_f[2] = const_cast<float *>(static_cast<const float *>(src[2]));
	} else {
		src_f[0] = static_cast<float *>(tmp) + 0 * n_aligned;
		src_f[1] = static_cast<float *>(tmp) + 1 * n_aligned;
		src_f[2] = static_cast<float *>(tmp) + 2 * n_aligned;
	}

	if (p->dst_fmt.type == timecube::PixelType::FLOAT) {
		dst_f[0] = const_cast<float *>(static_cast<const float *>(dst[0]));
		dst_f[1] = const_cast<float *>(static_cast<const float *>(dst[1]));
		dst_f[2] = const_cast<float *>(static_cast<const float *>(dst[2]));
	} else {
		dst_f[0] = static_cast<float *>(tmp) + 0 * n_aligned;
		dst_f[1] = static_cast<float *>(tmp) + 1 * n_aligned;
		dst_f[2] = static_cast<float *>(tmp) + 2 * n_aligned;
	}

	if (p->src_fmt.type != timecube::PixelType::FLOAT)
		lut->to_float(src, src_f, p->src_fmt, n);

	lut->process(src_f, dst_f, n);

	if (p->dst_fmt.type != timecube::PixelType::FLOAT)
		lut->from_float(dst_f, dst, p->dst_fmt, n);
} catch (...) {
		// ...
}

void timecube_filter_free(timecube_filter *ptr)
{
	delete static_cast<timecube::Lut *>(ptr);
}
