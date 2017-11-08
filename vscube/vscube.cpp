#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include "cube.h"
#include "lut.h"
#include "vsxx_pluginmain.h"

using namespace vsxx;

namespace {

timecube::PixelFormat vsformat_to_tcformat(const VSFormat &vsformat, const ConstPropertyMap &props)
{
	timecube::PixelFormat format{};

	if (vsformat.sampleType == stInteger && vsformat.bytesPerSample == 1)
		format.type = timecube::PixelType::BYTE;
	else if (vsformat.sampleType == stInteger && vsformat.bytesPerSample == 2)
		format.type = timecube::PixelType::WORD;
	else if (vsformat.sampleType == stFloat && vsformat.bytesPerSample == 2)
		format.type = timecube::PixelType::HALF;
	else if (vsformat.sampleType == stFloat && vsformat.bytesPerSample == 4)
		format.type = timecube::PixelType::FLOAT;
	else
		throw std::runtime_error{ "unknown pixel type" };

	format.depth = vsformat.bitsPerSample;
	format.fullrange = props.get_prop<int>("_ColorRange", map::default_val(0)) == 0;

	return format;
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
	std::unique_ptr<timecube::Lut> m_lut;

	void run_direct(const void * const src[3], void * const dst[3], const ptrdiff_t src_stride[3], const ptrdiff_t dst_stride[3], unsigned width, unsigned height) const
	{
		const float *src_p[3];
		float *dst_p[3];

		for (unsigned p = 0; p < 3; ++p) {
			src_p[p] = static_cast<const float *>(src[p]);
			dst_p[p] = static_cast<float *>(dst[p]);
		}

		for (unsigned i = 0; i < height; ++i) {
			m_lut->process(src_p, dst_p, width);

			for (unsigned p = 0; p < 3; ++p) {
				src_p[p] = increment(src_p[p], src_stride[p]);
				dst_p[p] = increment(dst_p[p], dst_stride[p]);
			}
		}
	}

	void run_indirect(const void * const src[3], void * const dst[3], const ptrdiff_t src_stride[3], const ptrdiff_t dst_stride[3],
	                  const timecube::PixelFormat &format, unsigned width, unsigned height) const
	{
		std::unique_ptr<float, decltype(&vs_aligned_free)> tmp_buf{ nullptr, vs_aligned_free };
		unsigned aligned_width = width % 8 ? (width - width % 8) + 8 : width;

		const void *src_p[3] = { src[0], src[1], src[2] };
		void *dst_p[3] = { dst[0], dst[1], dst[2] };
		float *tmp[3] = { 0 };

		tmp_buf.reset(vs_aligned_malloc<float>(aligned_width * 3 * sizeof(float), 32));
		if (!tmp_buf)
			throw std::bad_alloc{};

		tmp[0] = tmp_buf.get();
		tmp[1] = tmp_buf.get() + aligned_width;
		tmp[2] = tmp_buf.get() + aligned_width * 2;

		for (unsigned i = 0; i < height; ++i) {
			m_lut->to_float(src_p, tmp, format, width);
			m_lut->process(tmp, tmp, width);
			m_lut->from_float(tmp, dst_p, format, width);

			for (unsigned p = 0; p < 3; ++p) {
				src_p[p] = increment(src_p[p], src_stride[p]);
				dst_p[p] = increment(dst_p[p], dst_stride[p]);
			}
		}
	}

	void run(const ConstVideoFrame &src_frame, const VideoFrame &dst_frame, const timecube::PixelFormat &format, unsigned width, unsigned height) const
	{
		assert(static_cast<unsigned>(src_frame.width(0)) == width);
		assert(static_cast<unsigned>(src_frame.height(0)) == height);
		assert(static_cast<unsigned>(dst_frame.width(0)) == width);
		assert(static_cast<unsigned>(dst_frame.height(0)) == height);

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

		if (format.type == timecube::PixelType::FLOAT)
			run_direct(src_p, dst_p, src_stride, dst_stride, width, height);
		else
			run_indirect(src_p, dst_p, src_stride, dst_stride, format, width, height);
	}
public:
	explicit TimeCube(void *) {}

	const char *get_name(int) noexcept override { return "Cube"; }

	std::pair<::VSFilterMode, int> init(const ConstPropertyMap &in, const PropertyMap &out, const VapourCore &core) override
	{
		m_clip = in.get_prop<FilterNode>("clip");
		m_vi = m_clip.video_info();

		if (m_vi.format && m_vi.format->colorFamily != cmRGB)
			throw std::runtime_error{ "must be RGB" };
		if (m_vi.format && m_vi.format->sampleType == stInteger && m_vi.format->bitsPerSample > 16)
			throw std::runtime_error{ "more than 16-bit not supported" };

		const char *path = in.get_prop<const char *>("cube");
		int cpu = int64ToIntS(in.get_prop<int64_t>("cpu", map::default_val<int64_t>(INT_MAX)));

		timecube::Cube cube = timecube::read_cube_from_file(path);
		m_lut = timecube::create_lut_impl(cube, cpu);

		if (m_vi.format && m_vi.format->id == pfRGBH && !m_lut->supports_half())
			throw std::runtime_error{ "RGBH not supported on current CPU" };

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

		if (src_format.colorFamily != cmRGB)
			throw std::runtime_error{ "must be RGB" };

		timecube::PixelFormat format = vsformat_to_tcformat(src_format, src_frame.frame_props_ro());
		unsigned width = src_frame.width(0);
		unsigned height = src_frame.height(0);

		VideoFrame dst_frame = core.new_video_frame(src_format, width, height, src_frame);
		run(src_frame, dst_frame, format, width, height);

		return dst_frame;
	}
};

const PluginInfo g_plugin_info{
	"day.simultaneous.4", "timecube", "TimeCube 4D", {
		{ &vsxx::FilterBase::filter_create<TimeCube>, "Cube", "clip:clip;cube:data;cpu:int:opt;" }
	}
};
