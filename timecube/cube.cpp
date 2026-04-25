#include <cctype>
#include <cerrno>
#include <charconv>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include "cube.h"

#ifdef _WIN32
  #include <filesystem>
#endif

using namespace std::string_view_literals;

namespace timecube {
namespace {

// 5.3  A line of text shall not be longer than 250 bytes. Lines of text do not contain newline characters.
constexpr size_t LINE_LEN = 250 + 1 + 1;

struct FileCloser {
	void operator()(std::FILE *f) { std::fclose(f); }
};

[[noreturn]] void throw_system_error()
{
	throw std::system_error{ errno, std::system_category() };
}

void read_line(char *buf, FILE *f)
{
	do {
		if (!std::fgets(buf, LINE_LEN, f)) {
			if (std::feof(f))
				throw std::runtime_error{ "end of file" };
			else
				throw_system_error();
		}
	} while (buf[0] == '#' || buf[0] == '\n');
}

std::string_view skip_space(std::string_view buf)
{
	return buf.substr(std::min(buf.find_first_not_of(" \t"), buf.size()));
}

template <class T>
std::string_view parse_number(std::string_view buf, T *dst)
{
	std::size_t end_idx = std::min(buf.find_first_of(" \t\n"), buf.size());

	std::from_chars_result res = std::from_chars(&*buf.begin(), &*buf.begin() + end_idx, *dst);
	if (res.ec != std::error_code{} || res.ptr != &*buf.begin() + end_idx)
		throw std::runtime_error{ "invalid number" };

	return buf.substr(end_idx);
}

bool is_keyword(std::string_view buf, std::string_view keyword)
{
	size_t kw_size = keyword.size();
	return buf.size() > kw_size && buf.substr(0, kw_size) == keyword &&
		(buf[kw_size] == ' ' || buf[kw_size] == '\t' || buf[kw_size] == '\n');
}

std::string parse_title(std::string_view buf)
{
	buf = buf.substr("TITLE"sv.size());
	buf = skip_space(buf);

	if (buf.empty() || buf.front() != '"')
		throw std::runtime_error{ "missing opening quote in TITLE" };

	size_t end = buf.find('"', 1);
	if (end == std::string_view::npos)
		throw std::runtime_error{ "missing closing quote in TITLE" };

	return std::string{ buf.substr(1, end) };
}

void parse_domain_minmax(std::string_view buf, float dst[3])
{
	buf = buf.substr("DOMAIN_MIN"sv.size());
	buf = skip_space(buf);

	buf = parse_number(buf, dst + 0);
	buf = skip_space(buf);
	buf = parse_number(buf, dst + 1);
	buf = skip_space(buf);
	buf = parse_number(buf, dst + 2);
}

uint_least32_t parse_lut_size(std::string_view buf)
{
	uint_least32_t n;

	buf = buf.substr("LUT_1D_SIZE"sv.size());
	buf = skip_space(buf);
	buf = parse_number(buf, &n);

	return n;
}

void parse_lut_entry(std::string_view buf, float dst[3])
{
	buf = parse_number(buf, dst + 0);
	buf = skip_space(buf);
	buf = parse_number(buf, dst + 1);
	buf = skip_space(buf);
	buf = parse_number(buf, dst + 2);
}

size_t lut_size(uint_least32_t n, bool is_3d)
{
	uint_least32_t size = is_3d ? n * n * n : n;
#if UINT_LEAST32_MAX > SIZE_MAX
	if (size > SIZE_MAX)
		throw std::length_error{ "LUT exceeds memory capacity" };
#endif
	return size;
}

} // namespace


Cube read_cube_from_file(const char *path)
{
	Cube cube;
	std::unique_ptr<std::FILE, FileCloser> file_uptr;

#ifdef _WIN32
	file_uptr.reset(_wfopen(std::filesystem::u8path(path).c_str(), L"r"));
#else
	file_uptr.reset(std::fopen(path, "r"));
#endif
	std::FILE *file = file_uptr.get();
	char buf[LINE_LEN];
	std::string_view sv;

	if (!file)
		throw_system_error();

	// Headers.
	bool has_lut_size = false;

	while (true) {
		read_line(buf, file);
		sv = buf;

		if (is_keyword(sv, "TITLE")) {
			try {
				cube.title = parse_title(sv);
			} catch (...) {
				// Non-fatal.
			}
		} else if (is_keyword(sv, "DOMAIN_MIN")) {
			parse_domain_minmax(sv, cube.domain_min);
		} else if (is_keyword(sv, "DOMAIN_MAX")) {
			parse_domain_minmax(sv, cube.domain_max);
		} else if (is_keyword(sv, "LUT_1D_SIZE")) {
			if (has_lut_size)
				throw std::runtime_error{ "duplicate LUT declaration" };

			cube.n = parse_lut_size(sv);
			cube.is_3d = false;
			has_lut_size = true;
		} else if (is_keyword(sv, "LUT_3D_SIZE")) {
			if (has_lut_size)
				throw std::runtime_error{ "duplicate LUT declaration" };

			cube.n = parse_lut_size(sv);
			cube.is_3d = true;
			has_lut_size = true;
		} else if ("0123456789+-."sv.find(sv.front()) != std::string_view::npos) {
			break;
		}
	}
	if (!has_lut_size)
		throw std::runtime_error{ "missing LUT declaration" };

	if (cube.n < 2 || cube.n > (cube.is_3d ? 256U : 65536U))
		throw std::runtime_error{ "invalid LUT size" };
	if (cube.domain_min[0] > cube.domain_max[0] || cube.domain_min[1] > cube.domain_max[1] || cube.domain_min[2] > cube.domain_max[2])
		throw std::runtime_error{ "invalid domain" };

	// LUT.
	size_t size = lut_size(cube.n, cube.is_3d);

	cube.lut.insert(cube.lut.end(), 3, 0.0f);
	parse_lut_entry(sv, &*(cube.lut.end() - 3));

	for (unsigned i = 1; i < size; ++i) {
		read_line(buf, file);
		sv = buf;

		cube.lut.insert(cube.lut.end(), 3, 0.0f);
		parse_lut_entry(sv, &*(cube.lut.end() - 3));
	}

	return cube;
}

} // namespace timecube
