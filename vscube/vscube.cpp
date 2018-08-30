#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include "timecube.h"
#include "vsxx_pluginmain.h"

using namespace vsxx;

namespace {

struct TimecubeLutFree {
	void operator()(timecube_lut *ptr) { timecube_lut_free(ptr); }
};

struct TimecubeFilterFree {
	void operator()(timecube_filter *ptr) { timecube_filter_free(ptr); }
};

timecube_pixel_type_e vsformat_to_pixtype(const VSFormat &vsformat)
{
	if (vsformat.sampleType == stInteger && vsformat.bytesPerSample == 1)
		return TIMECUBE_PIXEL_BYTE;
	else if (vsformat.sampleType == stInteger && vsformat.bytesPerSample == 2)
		return TIMECUBE_PIXEL_WORD;
	else if (vsformat.sampleType == stFloat && vsformat.bytesPerSample == 2)
		return TIMECUBE_PIXEL_HALF;
	else if (vsformat.sampleType == stFloat && vsformat.bytesPerSample == 4)
		return TIMECUBE_PIXEL_FLOAT;
	else
		throw std::runtime_error{ "unknown pixel type" };
}

timecube_pixel_range_e props_to_range(const ConstPropertyMap &props)
{
	return props.get_prop<int>("_ColorRange", map::default_val(0)) == 0 ? TIMECUBE_RANGE_FULL : TIMECUBE_RANGE_LIMITED;
}

template <class T>
T *increment(T *ptr, ptrdiff_t count)
{
	return (T *)((const char *)ptr + count);
}

} // namespace


class TimeCube : public vsxx::FilterBase {
	FilterNode m_clip;
	::VSVideoInfo m_vi;
	std::unique_ptr<timecube_filter, TimecubeFilterFree> m_filter;
	timecube_pixel_range_e m_range;
	bool m_use_range;
public:
	explicit TimeCube(void *) : m_vi{}, m_range{}, m_use_range{} {}

	const char *get_name(int) noexcept override { return "Cube"; }

	std::pair<::VSFilterMode, int> init(const ConstPropertyMap &in, const PropertyMap &out, const VapourCore &core) override
	{
		m_clip = in.get_prop<FilterNode>("clip");
		::VSVideoInfo src_vi = m_clip.video_info();

		if (src_vi.format && src_vi.format->colorFamily != cmRGB)
			throw std::runtime_error{ "must be RGB" };

		m_vi = src_vi;
		if (in.contains("format")) {
			const ::VSFormat *format = core.format_preset(static_cast<::VSPresetFormat>(in.get_prop<int>("format")));
			if (!format)
				throw std::runtime_error{ "unregistered format" };
			if (format->colorFamily != cmRGB)
				throw std::runtime_error{ "must be RGB" };
			m_vi.format = format;
		}

		if (in.contains("range")) {
			m_range = static_cast<timecube_pixel_range_e>(in.get_prop<int>("range"));
			m_use_range = true;
		}

		const char *path = in.get_prop<const char *>("cube");
		int cpu = int64ToIntS(in.get_prop<int64_t>("cpu", map::default_val<int64_t>(INT64_MAX)));
		if (cpu < 0)
			cpu = INT_MAX;

		std::unique_ptr<timecube_lut, TimecubeLutFree> lut{ timecube_lut_from_file(path) };
		if (!lut)
			throw std::runtime_error{ "error reading LUT from file" };

		m_filter.reset(timecube_filter_create(lut.get(), static_cast<timecube_cpu_type_e>(cpu)));
		if (!m_filter)
			throw std::runtime_error{ "error creating LUT filter" };

		if (src_vi.format && !timecube_filter_supports_type(m_filter.get(), vsformat_to_pixtype(*src_vi.format)))
			throw std::runtime_error{ "input pixel type not supported" };
		if (m_vi.format && !timecube_filter_supports_type(m_filter.get(), vsformat_to_pixtype(*m_vi.format)))
			throw std::runtime_error{ "output pixel type not supported" };

		return{ fmParallel, 1 };
	}

	std::pair<const ::VSVideoInfo *, size_t> get_video_info() noexcept override
	{
		return{ &m_vi, 1 };
	}

	ConstVideoFrame get_frame_initial(int n, const VapourCore &core, ::VSFrameContext *frame_ctx) override
	{
		m_clip.request_frame_filter(n, frame_ctx);
		return ConstVideoFrame{};
	}

	ConstVideoFrame get_frame(int n, const VapourCore &core, ::VSFrameContext *frame_ctx) override
	{
		ConstVideoFrame src_frame = m_clip.get_frame_filter(n, frame_ctx);
		const VSFormat &src_format = src_frame.format();
		unsigned width = src_frame.width(0);
		unsigned height = src_frame.height(0);

		if (src_format.colorFamily != cmRGB)
			throw std::runtime_error{ "must be RGB" };

		if (!timecube_filter_supports_type(m_filter.get(), vsformat_to_pixtype(src_format)))
			throw std::runtime_error{ "pixel type not supported" };

		timecube_filter_params params{};
		params.src_type = vsformat_to_pixtype(src_format);
		params.src_depth = src_format.bitsPerSample;
		params.src_range = props_to_range(src_frame.frame_props_ro());
		params.dst_type = m_vi.format ? vsformat_to_pixtype(*m_vi.format) : params.src_type;
		params.dst_depth = m_vi.format ? m_vi.format->bitsPerSample : params.src_depth;
		params.dst_range = m_use_range ? m_range : params.src_range;

		timecube_filter_context ctx{};
		if (timecube_filter_create_context(m_filter.get(), &params, &ctx))
			throw std::runtime_error{ "error preparing filter" };

		VideoFrame dst_frame = core.new_video_frame(m_vi.format ? *m_vi.format : src_format, width, height, src_frame);
		std::unique_ptr<void, decltype(&vs_aligned_free)> tmp{ nullptr, vs_aligned_free };

		if (params.src_type != TIMECUBE_PIXEL_FLOAT || params.dst_type != TIMECUBE_PIXEL_FLOAT) {
			size_t n = ((width + 15) & ~15) * sizeof(float) * 3;
			tmp.reset(vs_aligned_malloc(n, 64));
			if (!tmp)
				throw std::runtime_error{ "error allocating buffer" };
		}

		const void *src_p[3];
		ptrdiff_t src_stride[3];

		void *dst_p[3];
		ptrdiff_t dst_stride[3];

		for (unsigned p = 0; p < 3; ++p) {
			src_p[p] = src_frame.read_ptr(p);
			src_stride[p] = src_frame.stride(p);

			dst_p[p] = dst_frame.write_ptr(p);
			dst_stride[p] = dst_frame.stride(p);
		}

		for (unsigned i = 0; i < height; ++i) {
			timecube_filter_context_apply(&ctx, src_p, dst_p, tmp.get(), width);

			for (unsigned p = 0; p < 3; ++p) {
				src_p[p] = increment(src_p[p], src_stride[p]);
				dst_p[p] = increment(dst_p[p], dst_stride[p]);
			}
		}

		return dst_frame;
	}
};

const PluginInfo g_plugin_info{
	"day.simultaneous.4", "timecube", "TimeCube 4D", {
		{ &vsxx::FilterBase::filter_create<TimeCube>, "Cube", "clip:clip;cube:data;format:int:opt;range:int:opt;cpu:int:opt;" }
	}
};
