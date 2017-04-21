CXX=clang++
SRC=$(shell find src -iname '*.cpp')
TARGETSRC=$(shell find targets -iname '*.cpp')
HEADERS=$(shell find include -iname '*.hpp')
OBJ=$(SRC:%.cpp=%.o)
TARGETOBJ=$(TARGETSRC:%.cpp=%.o)
mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(dir $(mkfile_path))
CXXFLAGS=-std=c++14 -I$(current_dir)include -g -DBOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
LDFLAGS=-framework OpenCL
TARGETS=$(TARGETSRC:targets/%.cpp=%)


all: $(TARGETS)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

targets/%.o: targets/%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%: targets/%.o $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^

.PHONY: clean all

clean:
	rm -f $(OBJ) $(TARGETS)
