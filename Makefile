MY_CFLAGS := -O2 -fPIC $(CFLAGS)
MY_CXXFLAGS := -std=c++14 -O2 -fPIC $(CXXFLAGS)
MY_CPPFLAGS := -Itimecube -Ivsxx $(CPPFLAGS)
MY_LDFLAGS := $(LDFLAGS)
MY_LIBS := $(LIBS)

timecube_HDRS = \
	timecube/cube.h \
	timecube/lut.h \
	timecube/timecube.h \
	timecube/x86/lut_x86.h

timecube_OBJS = \
	timecube/cube.o \
	timecube/lut.o \
	timecube/timecube.o \
	timecube/x86/lut_avx2.o \
	timecube/x86/lut_avx512.o \
	timecube/x86/lut_sse41.o \
	timecube/x86/lut_x86.o

vsxx_HDRS = \
	vsxx/VapourSynth.h \
	vsxx/VapourSynth++.hpp \
	vsxx/VSHelper.h \
	vsxx/VSScript.h \
	vsxx/vsxx_pluginmain.h

ifeq ($(X86), 1)
  timecube/x86/lut_avx2.o: EXTRA_CXXFLAGS := -mf16c -mavx2 -mfma -march=haswell
  timecube/x86/lut_sse41.o: EXTRA_CXXFLAGS := -msse4.1
  timecube/x86/lut_avx512.o: EXTRA_CXXFLAGS := -mf16c -mfma -mavx512f -mavx512cd -mavx512vl -mavx512bw -mavx512dq -mtune=skylake-avx512
  MY_CPPFLAGS := -DCUBE_X86 $(MY_CPPFLAGS)
endif

all: vscube.so

benchmark/benchmark: benchmark/main.o $(timecube_OBJS)
	$(CXX) $(MY_LDFLAGS) $^ $(MY_LIBS) -o $@

vscube.so: vscube/vscube.o vsxx/vsxx_pluginmain.o $(timecube_OBJS)
	$(CXX) -shared $(MY_LDFLAGS) $^ $(MY_LIBS) -o $@

clean:
	rm -f *.a *.o *.so benchmark/benchmark benchmark/*.o timecube/*.o timecube/x86/*.o vscube/*.o vsxx/*.o

%.o: %.cpp $(timecube_HDRS) $(vsxx_HDRS)
	$(CXX) -c $(EXTRA_CXXFLAGS) $(MY_CXXFLAGS) $(MY_CPPFLAGS) $< -o $@

.PHONY: clean
