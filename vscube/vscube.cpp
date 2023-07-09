#include <stdexcept>
#include "timecube.h"
#include "vsxx4_pluginmain.h"
#include "VSConstants4.h"
#include "VSHelper4.h"

using namespace vsxx4;

namespace {

struct TimecubeLutFree {
	void operator()(timecube_lut *ptr) { timecube_lut_free(ptr); }
};

struct TimecubeFilterFree {
	void operator()(timecube_filter *ptr) { timecube_filter_free(ptr); }
};

timecube_pixel_type_e vsformat_to_pixtype(const VSVideoFormat &vsformat)
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


class TimeCube : public FilterBase {
	FilterNode m_clip;
	VSVideoInfo m_vi;
	std::unique_ptr<timecube_filter, TimecubeFilterFree> m_filter_limited;
	std::unique_ptr<timecube_filter, TimecubeFilterFree> m_filter_full;
	timecube_pixel_range_e m_force_output_range;
public:
	TimeCube(void * = nullptr) : m_vi{}, m_force_output_range{ TIMECUBE_RANGE_INTERNAL } {}

	const char *get_name(void *) noexcept override { return "Cube"; }

	void init(const ConstMap &in, const Map &out, const Core &core) override
	{
		m_clip = in.get_prop<FilterNode>("clip");

		VSVideoInfo src_vi = m_clip.video_info();
		if (!vsh::isConstantVideoFormat(&src_vi))
			throw std::runtime_error{ "must be constant format" };
		if (src_vi.format.colorFamily != cfRGB)
			throw std::runtime_error{ "must be RGB" };

		m_vi = src_vi;
		if (in.contains("format")) {
			m_vi.format = core.get_video_format_by_id(in.get_prop<uint32_t>("format"));
			if (m_vi.format.colorFamily == cfUndefined)
				throw std::runtime_error{ "invalid format" };
			if (m_vi.format.colorFamily != cfRGB)
				throw std::runtime_error{ "must be RGB" };
		}

		const char *path = in.get_prop<const char *>("cube");
		std::unique_ptr<timecube_lut, TimecubeLutFree> lut{ timecube_lut_from_file(path) };
		if (!lut)
			throw std::runtime_error{ "error reading LUT from file" };

		int interp = vsh::int64ToIntS(in.get_prop<int64_t>("interp", map::default_val(0LL)));
		if (interp < 0)
			interp = 0;

		int cpu = vsh::int64ToIntS(in.get_prop<int64_t>("cpu", map::default_val(INT64_MAX)));
		if (cpu < 0)
			cpu = INT_MAX;

		if (in.contains("range")) {
			m_force_output_range = static_cast<timecube_pixel_range_e>(in.get_prop<int>("range"));
			if (m_force_output_range != TIMECUBE_RANGE_LIMITED && m_force_output_range != TIMECUBE_RANGE_FULL)
				throw std::runtime_error{ "range must be 0 (limited) or 1 (full)" };
		}

		timecube_filter_params params{};
		params.width = m_vi.width;
		params.height = m_vi.height;
		params.src_type = vsformat_to_pixtype(src_vi.format);
		params.src_depth = src_vi.format.bitsPerSample;
		params.src_range = TIMECUBE_RANGE_INTERNAL;
		params.dst_type = vsformat_to_pixtype(m_vi.format);
		params.dst_depth = m_vi.format.bitsPerSample;
		params.dst_range = m_force_output_range;
		params.interp = static_cast<timecube_interpolation_e>(interp);
		params.cpu = static_cast<timecube_cpu_type_e>(cpu);

		// Limited range input.
		params.src_range = TIMECUBE_RANGE_LIMITED;
		params.dst_range = m_force_output_range != TIMECUBE_RANGE_INTERNAL ? m_force_output_range : params.src_range;
		m_filter_limited.reset(timecube_filter_create(lut.get(), &params));
		if (!m_filter_limited)
			throw std::runtime_error{ "error creating filter" };

		// Full range input.
		params.src_range = TIMECUBE_RANGE_FULL;
		params.dst_range = m_force_output_range != TIMECUBE_RANGE_INTERNAL ? m_force_output_range : params.src_range;
		m_filter_full.reset(timecube_filter_create(lut.get(), &params));
		if (!m_filter_full)
			throw std::runtime_error{ "error creating filter" };

		create_video_filter(out, m_vi, fmParallel, simple_dep(m_clip, rpStrictSpatial), core);
	}

	ConstFrame get_frame_initial(int n, const Core &core, const FrameContext &frame_context, void *) override
	{
		frame_context.request_frame(n, m_clip);
		return nullptr;
	}

	ConstFrame get_frame(int n, const Core &core, const FrameContext &frame_context, void *) override
	{
		ConstFrame src_frame = frame_context.get_frame(n, m_clip);
		Frame dst_frame = core.new_video_frame(m_vi.format, m_vi.width, m_vi.height, src_frame);

		int vsrange = src_frame.frame_props_ro().get_prop<int>("_ColorRange", map::default_val(0));
		timecube_pixel_range_e range = vsrange == VSC_RANGE_FULL ? TIMECUBE_RANGE_FULL : TIMECUBE_RANGE_LIMITED;
		timecube_filter *filter = range == TIMECUBE_RANGE_FULL ? m_filter_full.get() : m_filter_limited.get();

		const void *srcp[3];
		ptrdiff_t src_stride[3];
		void *dstp[3];
		ptrdiff_t dst_stride[3];

		for (unsigned p = 0; p < 3; ++p) {
			srcp[p] = src_frame.read_ptr(p);
			src_stride[p] = src_frame.stride(p);
			dstp[p] = dst_frame.write_ptr(p);
			dst_stride[p] = dst_frame.stride(p);
		}

		std::unique_ptr<void, decltype(&vsh::vsh_aligned_free)> tmp{ nullptr, vsh::vsh_aligned_free };
		tmp.reset(vsh::vsh_aligned_malloc(timecube_filter_get_tmp_size(filter), 64));

		timecube_filter_apply(filter, srcp, src_stride, dstp, dst_stride, tmp.get());

		if (m_force_output_range != TIMECUBE_RANGE_INTERNAL) {
			VSColorRange output_range = m_force_output_range == TIMECUBE_RANGE_FULL ? VSC_RANGE_FULL : VSC_RANGE_LIMITED;
			dst_frame.frame_props_rw().set_prop("_ColorRange", static_cast<int>(output_range));
		}
		return dst_frame;
	}
};

} // namespace


const PluginInfo4 g_plugin_info4 = {
	"day.simultaneous.4", "timecube", "TimeCube 4D", 0, {
		{ &FilterBase::filter_create<TimeCube>, "Cube", "clip:vnode;cube:data;format:int:opt;range:int:opt;interp:int:opt;cpu:int:opt;", "clip:vnode;" }
	}
};
