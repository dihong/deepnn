# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl

# Include any dependencies generated for this target.
include CMakeFiles/proc.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/proc.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/proc.dir/flags.make

CMakeFiles/proc.dir/src/main.cpp.o: CMakeFiles/proc.dir/flags.make
CMakeFiles/proc.dir/src/main.cpp.o: src/main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/proc.dir/src/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/proc.dir/src/main.cpp.o -c /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/main.cpp

CMakeFiles/proc.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/proc.dir/src/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/main.cpp > CMakeFiles/proc.dir/src/main.cpp.i

CMakeFiles/proc.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/proc.dir/src/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/main.cpp -o CMakeFiles/proc.dir/src/main.cpp.s

CMakeFiles/proc.dir/src/main.cpp.o.requires:
.PHONY : CMakeFiles/proc.dir/src/main.cpp.o.requires

CMakeFiles/proc.dir/src/main.cpp.o.provides: CMakeFiles/proc.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/proc.dir/build.make CMakeFiles/proc.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/proc.dir/src/main.cpp.o.provides

CMakeFiles/proc.dir/src/main.cpp.o.provides.build: CMakeFiles/proc.dir/src/main.cpp.o

CMakeFiles/proc.dir/src/setup.cpp.o: CMakeFiles/proc.dir/flags.make
CMakeFiles/proc.dir/src/setup.cpp.o: src/setup.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/proc.dir/src/setup.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/proc.dir/src/setup.cpp.o -c /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/setup.cpp

CMakeFiles/proc.dir/src/setup.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/proc.dir/src/setup.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/setup.cpp > CMakeFiles/proc.dir/src/setup.cpp.i

CMakeFiles/proc.dir/src/setup.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/proc.dir/src/setup.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/setup.cpp -o CMakeFiles/proc.dir/src/setup.cpp.s

CMakeFiles/proc.dir/src/setup.cpp.o.requires:
.PHONY : CMakeFiles/proc.dir/src/setup.cpp.o.requires

CMakeFiles/proc.dir/src/setup.cpp.o.provides: CMakeFiles/proc.dir/src/setup.cpp.o.requires
	$(MAKE) -f CMakeFiles/proc.dir/build.make CMakeFiles/proc.dir/src/setup.cpp.o.provides.build
.PHONY : CMakeFiles/proc.dir/src/setup.cpp.o.provides

CMakeFiles/proc.dir/src/setup.cpp.o.provides.build: CMakeFiles/proc.dir/src/setup.cpp.o

CMakeFiles/proc.dir/src/common.cpp.o: CMakeFiles/proc.dir/flags.make
CMakeFiles/proc.dir/src/common.cpp.o: src/common.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/proc.dir/src/common.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/proc.dir/src/common.cpp.o -c /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/common.cpp

CMakeFiles/proc.dir/src/common.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/proc.dir/src/common.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/common.cpp > CMakeFiles/proc.dir/src/common.cpp.i

CMakeFiles/proc.dir/src/common.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/proc.dir/src/common.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/common.cpp -o CMakeFiles/proc.dir/src/common.cpp.s

CMakeFiles/proc.dir/src/common.cpp.o.requires:
.PHONY : CMakeFiles/proc.dir/src/common.cpp.o.requires

CMakeFiles/proc.dir/src/common.cpp.o.provides: CMakeFiles/proc.dir/src/common.cpp.o.requires
	$(MAKE) -f CMakeFiles/proc.dir/build.make CMakeFiles/proc.dir/src/common.cpp.o.provides.build
.PHONY : CMakeFiles/proc.dir/src/common.cpp.o.provides

CMakeFiles/proc.dir/src/common.cpp.o.provides.build: CMakeFiles/proc.dir/src/common.cpp.o

CMakeFiles/proc.dir/src/full.cpp.o: CMakeFiles/proc.dir/flags.make
CMakeFiles/proc.dir/src/full.cpp.o: src/full.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/proc.dir/src/full.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/proc.dir/src/full.cpp.o -c /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/full.cpp

CMakeFiles/proc.dir/src/full.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/proc.dir/src/full.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/full.cpp > CMakeFiles/proc.dir/src/full.cpp.i

CMakeFiles/proc.dir/src/full.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/proc.dir/src/full.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/full.cpp -o CMakeFiles/proc.dir/src/full.cpp.s

CMakeFiles/proc.dir/src/full.cpp.o.requires:
.PHONY : CMakeFiles/proc.dir/src/full.cpp.o.requires

CMakeFiles/proc.dir/src/full.cpp.o.provides: CMakeFiles/proc.dir/src/full.cpp.o.requires
	$(MAKE) -f CMakeFiles/proc.dir/build.make CMakeFiles/proc.dir/src/full.cpp.o.provides.build
.PHONY : CMakeFiles/proc.dir/src/full.cpp.o.provides

CMakeFiles/proc.dir/src/full.cpp.o.provides.build: CMakeFiles/proc.dir/src/full.cpp.o

CMakeFiles/proc.dir/src/full.forward.cpp.o: CMakeFiles/proc.dir/flags.make
CMakeFiles/proc.dir/src/full.forward.cpp.o: src/full.forward.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/CMakeFiles $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/proc.dir/src/full.forward.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/proc.dir/src/full.forward.cpp.o -c /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/full.forward.cpp

CMakeFiles/proc.dir/src/full.forward.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/proc.dir/src/full.forward.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/full.forward.cpp > CMakeFiles/proc.dir/src/full.forward.cpp.i

CMakeFiles/proc.dir/src/full.forward.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/proc.dir/src/full.forward.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/full.forward.cpp -o CMakeFiles/proc.dir/src/full.forward.cpp.s

CMakeFiles/proc.dir/src/full.forward.cpp.o.requires:
.PHONY : CMakeFiles/proc.dir/src/full.forward.cpp.o.requires

CMakeFiles/proc.dir/src/full.forward.cpp.o.provides: CMakeFiles/proc.dir/src/full.forward.cpp.o.requires
	$(MAKE) -f CMakeFiles/proc.dir/build.make CMakeFiles/proc.dir/src/full.forward.cpp.o.provides.build
.PHONY : CMakeFiles/proc.dir/src/full.forward.cpp.o.provides

CMakeFiles/proc.dir/src/full.forward.cpp.o.provides.build: CMakeFiles/proc.dir/src/full.forward.cpp.o

CMakeFiles/proc.dir/src/full.dEdX.cpp.o: CMakeFiles/proc.dir/flags.make
CMakeFiles/proc.dir/src/full.dEdX.cpp.o: src/full.dEdX.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/CMakeFiles $(CMAKE_PROGRESS_6)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/proc.dir/src/full.dEdX.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/proc.dir/src/full.dEdX.cpp.o -c /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/full.dEdX.cpp

CMakeFiles/proc.dir/src/full.dEdX.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/proc.dir/src/full.dEdX.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/full.dEdX.cpp > CMakeFiles/proc.dir/src/full.dEdX.cpp.i

CMakeFiles/proc.dir/src/full.dEdX.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/proc.dir/src/full.dEdX.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/full.dEdX.cpp -o CMakeFiles/proc.dir/src/full.dEdX.cpp.s

CMakeFiles/proc.dir/src/full.dEdX.cpp.o.requires:
.PHONY : CMakeFiles/proc.dir/src/full.dEdX.cpp.o.requires

CMakeFiles/proc.dir/src/full.dEdX.cpp.o.provides: CMakeFiles/proc.dir/src/full.dEdX.cpp.o.requires
	$(MAKE) -f CMakeFiles/proc.dir/build.make CMakeFiles/proc.dir/src/full.dEdX.cpp.o.provides.build
.PHONY : CMakeFiles/proc.dir/src/full.dEdX.cpp.o.provides

CMakeFiles/proc.dir/src/full.dEdX.cpp.o.provides.build: CMakeFiles/proc.dir/src/full.dEdX.cpp.o

CMakeFiles/proc.dir/src/full.dEdW.cpp.o: CMakeFiles/proc.dir/flags.make
CMakeFiles/proc.dir/src/full.dEdW.cpp.o: src/full.dEdW.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/CMakeFiles $(CMAKE_PROGRESS_7)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/proc.dir/src/full.dEdW.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/proc.dir/src/full.dEdW.cpp.o -c /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/full.dEdW.cpp

CMakeFiles/proc.dir/src/full.dEdW.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/proc.dir/src/full.dEdW.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/full.dEdW.cpp > CMakeFiles/proc.dir/src/full.dEdW.cpp.i

CMakeFiles/proc.dir/src/full.dEdW.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/proc.dir/src/full.dEdW.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/full.dEdW.cpp -o CMakeFiles/proc.dir/src/full.dEdW.cpp.s

CMakeFiles/proc.dir/src/full.dEdW.cpp.o.requires:
.PHONY : CMakeFiles/proc.dir/src/full.dEdW.cpp.o.requires

CMakeFiles/proc.dir/src/full.dEdW.cpp.o.provides: CMakeFiles/proc.dir/src/full.dEdW.cpp.o.requires
	$(MAKE) -f CMakeFiles/proc.dir/build.make CMakeFiles/proc.dir/src/full.dEdW.cpp.o.provides.build
.PHONY : CMakeFiles/proc.dir/src/full.dEdW.cpp.o.provides

CMakeFiles/proc.dir/src/full.dEdW.cpp.o.provides.build: CMakeFiles/proc.dir/src/full.dEdW.cpp.o

CMakeFiles/proc.dir/src/full.dEdB.cpp.o: CMakeFiles/proc.dir/flags.make
CMakeFiles/proc.dir/src/full.dEdB.cpp.o: src/full.dEdB.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/CMakeFiles $(CMAKE_PROGRESS_8)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/proc.dir/src/full.dEdB.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/proc.dir/src/full.dEdB.cpp.o -c /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/full.dEdB.cpp

CMakeFiles/proc.dir/src/full.dEdB.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/proc.dir/src/full.dEdB.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/full.dEdB.cpp > CMakeFiles/proc.dir/src/full.dEdB.cpp.i

CMakeFiles/proc.dir/src/full.dEdB.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/proc.dir/src/full.dEdB.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/full.dEdB.cpp -o CMakeFiles/proc.dir/src/full.dEdB.cpp.s

CMakeFiles/proc.dir/src/full.dEdB.cpp.o.requires:
.PHONY : CMakeFiles/proc.dir/src/full.dEdB.cpp.o.requires

CMakeFiles/proc.dir/src/full.dEdB.cpp.o.provides: CMakeFiles/proc.dir/src/full.dEdB.cpp.o.requires
	$(MAKE) -f CMakeFiles/proc.dir/build.make CMakeFiles/proc.dir/src/full.dEdB.cpp.o.provides.build
.PHONY : CMakeFiles/proc.dir/src/full.dEdB.cpp.o.provides

CMakeFiles/proc.dir/src/full.dEdB.cpp.o.provides.build: CMakeFiles/proc.dir/src/full.dEdB.cpp.o

CMakeFiles/proc.dir/src/softmax.cpp.o: CMakeFiles/proc.dir/flags.make
CMakeFiles/proc.dir/src/softmax.cpp.o: src/softmax.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/CMakeFiles $(CMAKE_PROGRESS_9)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/proc.dir/src/softmax.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/proc.dir/src/softmax.cpp.o -c /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/softmax.cpp

CMakeFiles/proc.dir/src/softmax.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/proc.dir/src/softmax.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/softmax.cpp > CMakeFiles/proc.dir/src/softmax.cpp.i

CMakeFiles/proc.dir/src/softmax.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/proc.dir/src/softmax.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/softmax.cpp -o CMakeFiles/proc.dir/src/softmax.cpp.s

CMakeFiles/proc.dir/src/softmax.cpp.o.requires:
.PHONY : CMakeFiles/proc.dir/src/softmax.cpp.o.requires

CMakeFiles/proc.dir/src/softmax.cpp.o.provides: CMakeFiles/proc.dir/src/softmax.cpp.o.requires
	$(MAKE) -f CMakeFiles/proc.dir/build.make CMakeFiles/proc.dir/src/softmax.cpp.o.provides.build
.PHONY : CMakeFiles/proc.dir/src/softmax.cpp.o.provides

CMakeFiles/proc.dir/src/softmax.cpp.o.provides.build: CMakeFiles/proc.dir/src/softmax.cpp.o

CMakeFiles/proc.dir/src/conv.cpp.o: CMakeFiles/proc.dir/flags.make
CMakeFiles/proc.dir/src/conv.cpp.o: src/conv.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/CMakeFiles $(CMAKE_PROGRESS_10)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/proc.dir/src/conv.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/proc.dir/src/conv.cpp.o -c /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/conv.cpp

CMakeFiles/proc.dir/src/conv.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/proc.dir/src/conv.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/conv.cpp > CMakeFiles/proc.dir/src/conv.cpp.i

CMakeFiles/proc.dir/src/conv.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/proc.dir/src/conv.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/conv.cpp -o CMakeFiles/proc.dir/src/conv.cpp.s

CMakeFiles/proc.dir/src/conv.cpp.o.requires:
.PHONY : CMakeFiles/proc.dir/src/conv.cpp.o.requires

CMakeFiles/proc.dir/src/conv.cpp.o.provides: CMakeFiles/proc.dir/src/conv.cpp.o.requires
	$(MAKE) -f CMakeFiles/proc.dir/build.make CMakeFiles/proc.dir/src/conv.cpp.o.provides.build
.PHONY : CMakeFiles/proc.dir/src/conv.cpp.o.provides

CMakeFiles/proc.dir/src/conv.cpp.o.provides.build: CMakeFiles/proc.dir/src/conv.cpp.o

CMakeFiles/proc.dir/src/schedule.cpp.o: CMakeFiles/proc.dir/flags.make
CMakeFiles/proc.dir/src/schedule.cpp.o: src/schedule.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/CMakeFiles $(CMAKE_PROGRESS_11)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/proc.dir/src/schedule.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/proc.dir/src/schedule.cpp.o -c /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/schedule.cpp

CMakeFiles/proc.dir/src/schedule.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/proc.dir/src/schedule.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/schedule.cpp > CMakeFiles/proc.dir/src/schedule.cpp.i

CMakeFiles/proc.dir/src/schedule.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/proc.dir/src/schedule.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/schedule.cpp -o CMakeFiles/proc.dir/src/schedule.cpp.s

CMakeFiles/proc.dir/src/schedule.cpp.o.requires:
.PHONY : CMakeFiles/proc.dir/src/schedule.cpp.o.requires

CMakeFiles/proc.dir/src/schedule.cpp.o.provides: CMakeFiles/proc.dir/src/schedule.cpp.o.requires
	$(MAKE) -f CMakeFiles/proc.dir/build.make CMakeFiles/proc.dir/src/schedule.cpp.o.provides.build
.PHONY : CMakeFiles/proc.dir/src/schedule.cpp.o.provides

CMakeFiles/proc.dir/src/schedule.cpp.o.provides.build: CMakeFiles/proc.dir/src/schedule.cpp.o

CMakeFiles/proc.dir/src/graph.cpp.o: CMakeFiles/proc.dir/flags.make
CMakeFiles/proc.dir/src/graph.cpp.o: src/graph.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/CMakeFiles $(CMAKE_PROGRESS_12)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/proc.dir/src/graph.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/proc.dir/src/graph.cpp.o -c /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/graph.cpp

CMakeFiles/proc.dir/src/graph.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/proc.dir/src/graph.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/graph.cpp > CMakeFiles/proc.dir/src/graph.cpp.i

CMakeFiles/proc.dir/src/graph.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/proc.dir/src/graph.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/graph.cpp -o CMakeFiles/proc.dir/src/graph.cpp.s

CMakeFiles/proc.dir/src/graph.cpp.o.requires:
.PHONY : CMakeFiles/proc.dir/src/graph.cpp.o.requires

CMakeFiles/proc.dir/src/graph.cpp.o.provides: CMakeFiles/proc.dir/src/graph.cpp.o.requires
	$(MAKE) -f CMakeFiles/proc.dir/build.make CMakeFiles/proc.dir/src/graph.cpp.o.provides.build
.PHONY : CMakeFiles/proc.dir/src/graph.cpp.o.provides

CMakeFiles/proc.dir/src/graph.cpp.o.provides.build: CMakeFiles/proc.dir/src/graph.cpp.o

CMakeFiles/proc.dir/src/input.cpp.o: CMakeFiles/proc.dir/flags.make
CMakeFiles/proc.dir/src/input.cpp.o: src/input.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/CMakeFiles $(CMAKE_PROGRESS_13)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/proc.dir/src/input.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/proc.dir/src/input.cpp.o -c /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/input.cpp

CMakeFiles/proc.dir/src/input.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/proc.dir/src/input.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/input.cpp > CMakeFiles/proc.dir/src/input.cpp.i

CMakeFiles/proc.dir/src/input.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/proc.dir/src/input.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/input.cpp -o CMakeFiles/proc.dir/src/input.cpp.s

CMakeFiles/proc.dir/src/input.cpp.o.requires:
.PHONY : CMakeFiles/proc.dir/src/input.cpp.o.requires

CMakeFiles/proc.dir/src/input.cpp.o.provides: CMakeFiles/proc.dir/src/input.cpp.o.requires
	$(MAKE) -f CMakeFiles/proc.dir/build.make CMakeFiles/proc.dir/src/input.cpp.o.provides.build
.PHONY : CMakeFiles/proc.dir/src/input.cpp.o.provides

CMakeFiles/proc.dir/src/input.cpp.o.provides.build: CMakeFiles/proc.dir/src/input.cpp.o

CMakeFiles/proc.dir/src/SOIL/SOIL.c.o: CMakeFiles/proc.dir/flags.make
CMakeFiles/proc.dir/src/SOIL/SOIL.c.o: src/SOIL/SOIL.c
	$(CMAKE_COMMAND) -E cmake_progress_report /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/CMakeFiles $(CMAKE_PROGRESS_14)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/proc.dir/src/SOIL/SOIL.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/proc.dir/src/SOIL/SOIL.c.o   -c /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/SOIL/SOIL.c

CMakeFiles/proc.dir/src/SOIL/SOIL.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/proc.dir/src/SOIL/SOIL.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/SOIL/SOIL.c > CMakeFiles/proc.dir/src/SOIL/SOIL.c.i

CMakeFiles/proc.dir/src/SOIL/SOIL.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/proc.dir/src/SOIL/SOIL.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/SOIL/SOIL.c -o CMakeFiles/proc.dir/src/SOIL/SOIL.c.s

CMakeFiles/proc.dir/src/SOIL/SOIL.c.o.requires:
.PHONY : CMakeFiles/proc.dir/src/SOIL/SOIL.c.o.requires

CMakeFiles/proc.dir/src/SOIL/SOIL.c.o.provides: CMakeFiles/proc.dir/src/SOIL/SOIL.c.o.requires
	$(MAKE) -f CMakeFiles/proc.dir/build.make CMakeFiles/proc.dir/src/SOIL/SOIL.c.o.provides.build
.PHONY : CMakeFiles/proc.dir/src/SOIL/SOIL.c.o.provides

CMakeFiles/proc.dir/src/SOIL/SOIL.c.o.provides.build: CMakeFiles/proc.dir/src/SOIL/SOIL.c.o

CMakeFiles/proc.dir/src/SOIL/stb_image_aug.c.o: CMakeFiles/proc.dir/flags.make
CMakeFiles/proc.dir/src/SOIL/stb_image_aug.c.o: src/SOIL/stb_image_aug.c
	$(CMAKE_COMMAND) -E cmake_progress_report /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/CMakeFiles $(CMAKE_PROGRESS_15)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/proc.dir/src/SOIL/stb_image_aug.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/proc.dir/src/SOIL/stb_image_aug.c.o   -c /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/SOIL/stb_image_aug.c

CMakeFiles/proc.dir/src/SOIL/stb_image_aug.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/proc.dir/src/SOIL/stb_image_aug.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/SOIL/stb_image_aug.c > CMakeFiles/proc.dir/src/SOIL/stb_image_aug.c.i

CMakeFiles/proc.dir/src/SOIL/stb_image_aug.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/proc.dir/src/SOIL/stb_image_aug.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/SOIL/stb_image_aug.c -o CMakeFiles/proc.dir/src/SOIL/stb_image_aug.c.s

CMakeFiles/proc.dir/src/SOIL/stb_image_aug.c.o.requires:
.PHONY : CMakeFiles/proc.dir/src/SOIL/stb_image_aug.c.o.requires

CMakeFiles/proc.dir/src/SOIL/stb_image_aug.c.o.provides: CMakeFiles/proc.dir/src/SOIL/stb_image_aug.c.o.requires
	$(MAKE) -f CMakeFiles/proc.dir/build.make CMakeFiles/proc.dir/src/SOIL/stb_image_aug.c.o.provides.build
.PHONY : CMakeFiles/proc.dir/src/SOIL/stb_image_aug.c.o.provides

CMakeFiles/proc.dir/src/SOIL/stb_image_aug.c.o.provides.build: CMakeFiles/proc.dir/src/SOIL/stb_image_aug.c.o

CMakeFiles/proc.dir/src/SOIL/image_helper.c.o: CMakeFiles/proc.dir/flags.make
CMakeFiles/proc.dir/src/SOIL/image_helper.c.o: src/SOIL/image_helper.c
	$(CMAKE_COMMAND) -E cmake_progress_report /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/CMakeFiles $(CMAKE_PROGRESS_16)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/proc.dir/src/SOIL/image_helper.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/proc.dir/src/SOIL/image_helper.c.o   -c /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/SOIL/image_helper.c

CMakeFiles/proc.dir/src/SOIL/image_helper.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/proc.dir/src/SOIL/image_helper.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/SOIL/image_helper.c > CMakeFiles/proc.dir/src/SOIL/image_helper.c.i

CMakeFiles/proc.dir/src/SOIL/image_helper.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/proc.dir/src/SOIL/image_helper.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/SOIL/image_helper.c -o CMakeFiles/proc.dir/src/SOIL/image_helper.c.s

CMakeFiles/proc.dir/src/SOIL/image_helper.c.o.requires:
.PHONY : CMakeFiles/proc.dir/src/SOIL/image_helper.c.o.requires

CMakeFiles/proc.dir/src/SOIL/image_helper.c.o.provides: CMakeFiles/proc.dir/src/SOIL/image_helper.c.o.requires
	$(MAKE) -f CMakeFiles/proc.dir/build.make CMakeFiles/proc.dir/src/SOIL/image_helper.c.o.provides.build
.PHONY : CMakeFiles/proc.dir/src/SOIL/image_helper.c.o.provides

CMakeFiles/proc.dir/src/SOIL/image_helper.c.o.provides.build: CMakeFiles/proc.dir/src/SOIL/image_helper.c.o

CMakeFiles/proc.dir/src/SOIL/image_DXT.c.o: CMakeFiles/proc.dir/flags.make
CMakeFiles/proc.dir/src/SOIL/image_DXT.c.o: src/SOIL/image_DXT.c
	$(CMAKE_COMMAND) -E cmake_progress_report /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/CMakeFiles $(CMAKE_PROGRESS_17)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/proc.dir/src/SOIL/image_DXT.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/proc.dir/src/SOIL/image_DXT.c.o   -c /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/SOIL/image_DXT.c

CMakeFiles/proc.dir/src/SOIL/image_DXT.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/proc.dir/src/SOIL/image_DXT.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/SOIL/image_DXT.c > CMakeFiles/proc.dir/src/SOIL/image_DXT.c.i

CMakeFiles/proc.dir/src/SOIL/image_DXT.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/proc.dir/src/SOIL/image_DXT.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/SOIL/image_DXT.c -o CMakeFiles/proc.dir/src/SOIL/image_DXT.c.s

CMakeFiles/proc.dir/src/SOIL/image_DXT.c.o.requires:
.PHONY : CMakeFiles/proc.dir/src/SOIL/image_DXT.c.o.requires

CMakeFiles/proc.dir/src/SOIL/image_DXT.c.o.provides: CMakeFiles/proc.dir/src/SOIL/image_DXT.c.o.requires
	$(MAKE) -f CMakeFiles/proc.dir/build.make CMakeFiles/proc.dir/src/SOIL/image_DXT.c.o.provides.build
.PHONY : CMakeFiles/proc.dir/src/SOIL/image_DXT.c.o.provides

CMakeFiles/proc.dir/src/SOIL/image_DXT.c.o.provides.build: CMakeFiles/proc.dir/src/SOIL/image_DXT.c.o

# Object files for target proc
proc_OBJECTS = \
"CMakeFiles/proc.dir/src/main.cpp.o" \
"CMakeFiles/proc.dir/src/setup.cpp.o" \
"CMakeFiles/proc.dir/src/common.cpp.o" \
"CMakeFiles/proc.dir/src/full.cpp.o" \
"CMakeFiles/proc.dir/src/full.forward.cpp.o" \
"CMakeFiles/proc.dir/src/full.dEdX.cpp.o" \
"CMakeFiles/proc.dir/src/full.dEdW.cpp.o" \
"CMakeFiles/proc.dir/src/full.dEdB.cpp.o" \
"CMakeFiles/proc.dir/src/softmax.cpp.o" \
"CMakeFiles/proc.dir/src/conv.cpp.o" \
"CMakeFiles/proc.dir/src/schedule.cpp.o" \
"CMakeFiles/proc.dir/src/graph.cpp.o" \
"CMakeFiles/proc.dir/src/input.cpp.o" \
"CMakeFiles/proc.dir/src/SOIL/SOIL.c.o" \
"CMakeFiles/proc.dir/src/SOIL/stb_image_aug.c.o" \
"CMakeFiles/proc.dir/src/SOIL/image_helper.c.o" \
"CMakeFiles/proc.dir/src/SOIL/image_DXT.c.o"

# External object files for target proc
proc_EXTERNAL_OBJECTS =

bin/x86_64/Release/proc: CMakeFiles/proc.dir/src/main.cpp.o
bin/x86_64/Release/proc: CMakeFiles/proc.dir/src/setup.cpp.o
bin/x86_64/Release/proc: CMakeFiles/proc.dir/src/common.cpp.o
bin/x86_64/Release/proc: CMakeFiles/proc.dir/src/full.cpp.o
bin/x86_64/Release/proc: CMakeFiles/proc.dir/src/full.forward.cpp.o
bin/x86_64/Release/proc: CMakeFiles/proc.dir/src/full.dEdX.cpp.o
bin/x86_64/Release/proc: CMakeFiles/proc.dir/src/full.dEdW.cpp.o
bin/x86_64/Release/proc: CMakeFiles/proc.dir/src/full.dEdB.cpp.o
bin/x86_64/Release/proc: CMakeFiles/proc.dir/src/softmax.cpp.o
bin/x86_64/Release/proc: CMakeFiles/proc.dir/src/conv.cpp.o
bin/x86_64/Release/proc: CMakeFiles/proc.dir/src/schedule.cpp.o
bin/x86_64/Release/proc: CMakeFiles/proc.dir/src/graph.cpp.o
bin/x86_64/Release/proc: CMakeFiles/proc.dir/src/input.cpp.o
bin/x86_64/Release/proc: CMakeFiles/proc.dir/src/SOIL/SOIL.c.o
bin/x86_64/Release/proc: CMakeFiles/proc.dir/src/SOIL/stb_image_aug.c.o
bin/x86_64/Release/proc: CMakeFiles/proc.dir/src/SOIL/image_helper.c.o
bin/x86_64/Release/proc: CMakeFiles/proc.dir/src/SOIL/image_DXT.c.o
bin/x86_64/Release/proc: CMakeFiles/proc.dir/build.make
bin/x86_64/Release/proc: /usr/lib/x86_64-linux-gnu/libOpenCL.so
bin/x86_64/Release/proc: lib/clFFT/libclFFT.so
bin/x86_64/Release/proc: CMakeFiles/proc.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable bin/x86_64/Release/proc"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/proc.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/proc.dir/build: bin/x86_64/Release/proc
.PHONY : CMakeFiles/proc.dir/build

CMakeFiles/proc.dir/requires: CMakeFiles/proc.dir/src/main.cpp.o.requires
CMakeFiles/proc.dir/requires: CMakeFiles/proc.dir/src/setup.cpp.o.requires
CMakeFiles/proc.dir/requires: CMakeFiles/proc.dir/src/common.cpp.o.requires
CMakeFiles/proc.dir/requires: CMakeFiles/proc.dir/src/full.cpp.o.requires
CMakeFiles/proc.dir/requires: CMakeFiles/proc.dir/src/full.forward.cpp.o.requires
CMakeFiles/proc.dir/requires: CMakeFiles/proc.dir/src/full.dEdX.cpp.o.requires
CMakeFiles/proc.dir/requires: CMakeFiles/proc.dir/src/full.dEdW.cpp.o.requires
CMakeFiles/proc.dir/requires: CMakeFiles/proc.dir/src/full.dEdB.cpp.o.requires
CMakeFiles/proc.dir/requires: CMakeFiles/proc.dir/src/softmax.cpp.o.requires
CMakeFiles/proc.dir/requires: CMakeFiles/proc.dir/src/conv.cpp.o.requires
CMakeFiles/proc.dir/requires: CMakeFiles/proc.dir/src/schedule.cpp.o.requires
CMakeFiles/proc.dir/requires: CMakeFiles/proc.dir/src/graph.cpp.o.requires
CMakeFiles/proc.dir/requires: CMakeFiles/proc.dir/src/input.cpp.o.requires
CMakeFiles/proc.dir/requires: CMakeFiles/proc.dir/src/SOIL/SOIL.c.o.requires
CMakeFiles/proc.dir/requires: CMakeFiles/proc.dir/src/SOIL/stb_image_aug.c.o.requires
CMakeFiles/proc.dir/requires: CMakeFiles/proc.dir/src/SOIL/image_helper.c.o.requires
CMakeFiles/proc.dir/requires: CMakeFiles/proc.dir/src/SOIL/image_DXT.c.o.requires
.PHONY : CMakeFiles/proc.dir/requires

CMakeFiles/proc.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/proc.dir/cmake_clean.cmake
.PHONY : CMakeFiles/proc.dir/clean

CMakeFiles/proc.dir/depend:
	cd /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/CMakeFiles/proc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/proc.dir/depend
