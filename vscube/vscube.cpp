#include <cassert>
#include <memory>
#include <stdexcept>
#include "cube.h"
#include "lut.h"
#include "vsxx_pluginmain.h"

using namespace vsxx;

class TimeCube : public vsxx::FilterBase {
	FilterNode m_clip;
	::VSVideoInfo m_vi;
	std::unique_ptr<timecube::Lut> m_lut;
public:
	explicit TimeCube(void *) {}

	const char *get_name(int) noexcept override { return "Cube"; }

	std::pair<::VSFilterMode, int> init(const ConstPropertyMap &in, const PropertyMap &out, const VapourCore &core) override
	{
		m_clip = in.get_prop<FilterNode>("clip");
		m_vi = m_clip.video_info();

		if (!isConstantFormat(&m_vi))
			throw std::runtime_error{ "must be constant format" };
		if (m_vi.format->colorFamily != cmRGB)
			throw std::runtime_error{ "must be RGB" };
		if (m_vi.format->id != pfRGBS)
			throw std::runtime_error{ "must be RGBS" };

		const char *path = in.get_prop<const char *>("cube");
		bool cpu = in.get_prop<bool>("cpu", map::default_val(true));

		timecube::Cube cube = timecube::read_cube_from_file(path);
		m_lut = timecube::create_lut_impl(cube, cpu);

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
		VideoFrame dst_frame = core.new_video_frame(*m_vi.format, m_vi.width, m_vi.height, src_frame);

		unsigned width = m_vi.width;
		unsigned height = m_vi.height;

		assert(width == src_frame.width(0));
		assert(height == src_frame.height(0));

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
			m_lut->process(src_p, dst_p, width);

			for (unsigned p = 0; p < 3; ++p) {
				src_p[p] = static_cast<const char *>(src_p[p]) + src_stride[p];
				dst_p[p] = static_cast<char *>(dst_p[p]) + dst_stride[p];
			}
		}

		return dst_frame;
	}
};

const PluginInfo g_plugin_info{
	"day.simultaneous.4", "timecube", "TimeCube 4D", {
		{ &vsxx::FilterBase::filter_create<TimeCube>, "Cube", "clip:clip;cube:data;cpu:int:opt;" }
	}
};
