#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>
#include <graphengine/graph.h>
#include <graphengine/filter.h>
#include "cube.h"
#include "lut.h"
#include "timecube.h"

#ifdef _WIN32
  #include <Windows.h>
#endif

struct timecube_filter {
protected:
	~timecube_filter() = default;
};

namespace {

struct FileCloser {
	void operator()(std::FILE *file) const { std::fclose(file); }
};

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


class TimecubeFilterGraph : public timecube_filter {
	std::vector<std::unique_ptr<graphengine::Filter>> m_filters;
	graphengine::GraphImpl m_graph;
	graphengine::node_id m_source_id = graphengine::null_node;
	graphengine::node_id m_sink_id = graphengine::null_node;
public:
	TimecubeFilterGraph(const timecube::Cube &cube, unsigned width, unsigned height, const timecube::PixelFormat &src_format, const timecube::PixelFormat &dst_format, int cpu)
	{
		graphengine::PlaneDescriptor format[3];
		std::fill_n(format, 3, graphengine::PlaneDescriptor{ width, height, timecube::pixel_size(src_format.type) });
		m_source_id = m_graph.add_source(3, format);

		graphengine::node_dep_desc ids[3] = { { m_source_id, 0 }, { m_source_id, 1 }, { m_source_id, 2 } };

		if (src_format.type != timecube::PixelType::FLOAT) {
			std::unique_ptr<graphengine::Filter> to_float_filter = timecube::create_to_float_impl(width, height, src_format, cpu);
			ids[0] = { m_graph.add_transform(to_float_filter.get(), &ids[0]), 0 };
			ids[1] = { m_graph.add_transform(to_float_filter.get(), &ids[1]), 0 };
			ids[2] = { m_graph.add_transform(to_float_filter.get(), &ids[2]), 0 };
			m_filters.push_back(std::move(to_float_filter));
		}

		if (cube.is_3d) {
			std::unique_ptr<graphengine::Filter> lut_filter = timecube::create_lut3d_impl(cube, width, height, cpu);
			graphengine::node_id id = m_graph.add_transform(lut_filter.get(), ids);
			ids[0] = { id, 0 };
			ids[1] = { id, 1 };
			ids[2] = { id, 2 };
			m_filters.push_back(std::move(lut_filter));
		} else {
			for (unsigned p = 0; p < 3; ++p) {
				std::unique_ptr<graphengine::Filter> lut_filter = timecube::create_lut1d_impl(cube, width, height, p, cpu);
				ids[p] = { m_graph.add_transform(lut_filter.get(), &ids[p]), 0 };
				m_filters.push_back(std::move(lut_filter));
			}
		}

		if (dst_format.type != timecube::PixelType::FLOAT) {
			std::unique_ptr<graphengine::Filter> from_float_filter = timecube::create_from_float_impl(width, height, dst_format, cpu);
			ids[0] = { m_graph.add_transform(from_float_filter.get(), &ids[0]), 0 };
			ids[1] = { m_graph.add_transform(from_float_filter.get(), &ids[1]), 0 };
			ids[2] = { m_graph.add_transform(from_float_filter.get(), &ids[2]), 0 };
			m_filters.push_back(std::move(from_float_filter));
		}

		m_sink_id = m_graph.add_sink(3, ids);
	}

	size_t get_tmp_size() const { return m_graph.get_tmp_size(false); }

	void apply(const void * const src[3], const ptrdiff_t src_stride[3], void * const dst[3], const ptrdiff_t dst_stride[3], void *tmp) const
	{
		graphengine::BufferDescriptor src_buffer[] = {
			{ const_cast<void *>(src[0]), src_stride[0], graphengine::BUFFER_MAX },
			{ const_cast<void *>(src[1]), src_stride[1], graphengine::BUFFER_MAX },
			{ const_cast<void *>(src[2]), src_stride[2], graphengine::BUFFER_MAX },
		};

		graphengine::BufferDescriptor dst_buffer[] = {
			{ dst[0], dst_stride[0], graphengine::BUFFER_MAX },
			{ dst[1], dst_stride[1], graphengine::BUFFER_MAX },
			{ dst[2], dst_stride[2], graphengine::BUFFER_MAX },
		};

		graphengine::Graph::Endpoint endpoints[] = {
			{ m_source_id, src_buffer },
			{ m_sink_id, dst_buffer },
		};

		m_graph.run(endpoints, tmp);
	}
};

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

timecube_filter *timecube_filter_create(const timecube_lut *lut, const timecube_filter_params *params, unsigned width, unsigned height, timecube_cpu_type_e cpu) try
{
	const timecube::Cube *cube = static_cast<const timecube::Cube *>(lut);
	timecube::PixelFormat src_format{ static_cast<timecube::PixelType>(params->src_type), params->src_depth, params->src_range == TIMECUBE_RANGE_FULL };
	timecube::PixelFormat dst_format{ static_cast<timecube::PixelType>(params->dst_type), params->dst_depth, params->dst_range == TIMECUBE_RANGE_FULL };

	std::unique_ptr<TimecubeFilterGraph> filter = std::make_unique<TimecubeFilterGraph>(*cube, width, height, src_format, dst_format, cpu);
	return filter.release();
} catch (...) {
	return nullptr;
}

size_t timecube_filter_get_tmp_size(const timecube_filter *filter) noexcept
{
	return static_cast<const TimecubeFilterGraph *>(filter)->get_tmp_size();
}

void timecube_filter_apply(const timecube_filter *filter, const void * const src[3], const ptrdiff_t src_stride[3], void * const dst[3], const ptrdiff_t dst_stride[3], void *tmp) noexcept
{
	static_cast<const TimecubeFilterGraph *>(filter)->apply(src, src_stride, dst, dst_stride, tmp);
}

void timecube_filter_free(timecube_filter *ptr)
{
	delete static_cast<TimecubeFilterGraph *>(ptr);
}
