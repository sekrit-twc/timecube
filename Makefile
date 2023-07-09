MY_CFLAGS := -O2 -fPIC $(CFLAGS)
MY_CXXFLAGS := -std=c++14 -O2 -fPIC -fvisibility=hidden $(CXXFLAGS)
MY_CPPFLAGS := -DGRAPHENGINE_IMPL_NAMESPACE=timecube -DNDEBUG -Igraphengine/include -Itimecube -Ivsxx -Ivsxx/vapoursynth $(CPPFLAGS)
MY_LDFLAGS := $(LDFLAGS)
MY_LIBS := $(LIBS)

graphengine_HDRS = \
        graphengine/graphengine/cpuinfo.h \
        graphengine/graphengine/node.h \
        graphengine/graphengine/state.h \
        graphengine/graphengine/x86/cpuinfo_x86.h \
        graphengine/include/graphengine/filter.h \
        graphengine/include/graphengine/graph.h \
        graphengine/include/graphengine/namespace.h \
        graphengine/include/graphengine/types.h

graphengine_OBJS = \
        graphengine/graphengine/cpuinfo.o \
        graphengine/graphengine/graph.o \
        graphengine/graphengine/node.o \
        graphengine/graphengine/x86/cpuinfo_x86.o

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
	vsxx/vapoursynth/VapourSynth.h \
	vsxx/vapoursynth/VSConstants4.h \
	vsxx/vapoursynth/VSHelper4.h \
	vsxx/VapourSynth4++.hpp \
	vsxx/vsxx4_pluginmain.h

ifeq ($(X86), 1)
  timecube/x86/lut_avx2.o: EXTRA_CXXFLAGS := -mf16c -mavx2 -mfma -march=haswell
  timecube/x86/lut_sse41.o: EXTRA_CXXFLAGS := -msse4.1
  timecube/x86/lut_avx512.o: EXTRA_CXXFLAGS := -mf16c -mfma -mavx512f -mavx512cd -mavx512vl -mavx512bw -mavx512dq -mtune=skylake-avx512
  MY_CPPFLAGS := -DCUBE_X86 $(MY_CPPFLAGS)
endif

all: vscube.so

benchmark/benchmark: benchmark/main.o $(timecube_OBJS) $(graphengine_OBJS)
	$(CXX) $(MY_LDFLAGS) $^ $(MY_LIBS) -o $@

vscube.so: vscube/vscube.o vsxx/vsxx4_pluginmain.o $(timecube_OBJS) $(graphengine_OBJS)
	$(CXX) -shared $(MY_LDFLAGS) $^ $(MY_LIBS) -o $@

clean:
	rm -f *.a *.o *.so benchmark/benchmark benchmark/*.o graphengine/graphengine/*.o graphengine/graphengine/x86/*.o timecube/*.o timecube/x86/*.o vscube/*.o vsxx/*.o

%.o: %.cpp $(graphengine_HDRS) $(timecube_HDRS) $(vsxx_HDRS)
	$(CXX) -c $(EXTRA_CXXFLAGS) $(MY_CXXFLAGS) $(MY_CPPFLAGS) $< -o $@

.PHONY: clean
