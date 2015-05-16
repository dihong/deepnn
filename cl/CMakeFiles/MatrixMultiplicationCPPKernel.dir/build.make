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
include CMakeFiles/MatrixMultiplicationCPPKernel.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/MatrixMultiplicationCPPKernel.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/MatrixMultiplicationCPPKernel.dir/flags.make

CMakeFiles/MatrixMultiplicationCPPKernel.dir/src/MatrixMultiplicationCPPKernel.cpp.o: CMakeFiles/MatrixMultiplicationCPPKernel.dir/flags.make
CMakeFiles/MatrixMultiplicationCPPKernel.dir/src/MatrixMultiplicationCPPKernel.cpp.o: src/MatrixMultiplicationCPPKernel.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/MatrixMultiplicationCPPKernel.dir/src/MatrixMultiplicationCPPKernel.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/MatrixMultiplicationCPPKernel.dir/src/MatrixMultiplicationCPPKernel.cpp.o -c /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/MatrixMultiplicationCPPKernel.cpp

CMakeFiles/MatrixMultiplicationCPPKernel.dir/src/MatrixMultiplicationCPPKernel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MatrixMultiplicationCPPKernel.dir/src/MatrixMultiplicationCPPKernel.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/MatrixMultiplicationCPPKernel.cpp > CMakeFiles/MatrixMultiplicationCPPKernel.dir/src/MatrixMultiplicationCPPKernel.cpp.i

CMakeFiles/MatrixMultiplicationCPPKernel.dir/src/MatrixMultiplicationCPPKernel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MatrixMultiplicationCPPKernel.dir/src/MatrixMultiplicationCPPKernel.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/MatrixMultiplicationCPPKernel.cpp -o CMakeFiles/MatrixMultiplicationCPPKernel.dir/src/MatrixMultiplicationCPPKernel.cpp.s

CMakeFiles/MatrixMultiplicationCPPKernel.dir/src/MatrixMultiplicationCPPKernel.cpp.o.requires:
.PHONY : CMakeFiles/MatrixMultiplicationCPPKernel.dir/src/MatrixMultiplicationCPPKernel.cpp.o.requires

CMakeFiles/MatrixMultiplicationCPPKernel.dir/src/MatrixMultiplicationCPPKernel.cpp.o.provides: CMakeFiles/MatrixMultiplicationCPPKernel.dir/src/MatrixMultiplicationCPPKernel.cpp.o.requires
	$(MAKE) -f CMakeFiles/MatrixMultiplicationCPPKernel.dir/build.make CMakeFiles/MatrixMultiplicationCPPKernel.dir/src/MatrixMultiplicationCPPKernel.cpp.o.provides.build
.PHONY : CMakeFiles/MatrixMultiplicationCPPKernel.dir/src/MatrixMultiplicationCPPKernel.cpp.o.provides

CMakeFiles/MatrixMultiplicationCPPKernel.dir/src/MatrixMultiplicationCPPKernel.cpp.o.provides.build: CMakeFiles/MatrixMultiplicationCPPKernel.dir/src/MatrixMultiplicationCPPKernel.cpp.o

# Object files for target MatrixMultiplicationCPPKernel
MatrixMultiplicationCPPKernel_OBJECTS = \
"CMakeFiles/MatrixMultiplicationCPPKernel.dir/src/MatrixMultiplicationCPPKernel.cpp.o"

# External object files for target MatrixMultiplicationCPPKernel
MatrixMultiplicationCPPKernel_EXTERNAL_OBJECTS =

bin/x86_64/Release/MatrixMultiplicationCPPKernel: CMakeFiles/MatrixMultiplicationCPPKernel.dir/src/MatrixMultiplicationCPPKernel.cpp.o
bin/x86_64/Release/MatrixMultiplicationCPPKernel: CMakeFiles/MatrixMultiplicationCPPKernel.dir/build.make
bin/x86_64/Release/MatrixMultiplicationCPPKernel: /usr/lib/x86_64-linux-gnu/libOpenCL.so
bin/x86_64/Release/MatrixMultiplicationCPPKernel: CMakeFiles/MatrixMultiplicationCPPKernel.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable bin/x86_64/Release/MatrixMultiplicationCPPKernel"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MatrixMultiplicationCPPKernel.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/cmake -E copy_if_different /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/MatrixMultiplicationCPPKernel_Kernels.cl /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/bin/x86_64/Release/.
	/usr/bin/cmake -E copy_if_different /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/src/MatrixMultiplicationCPPKernel_Kernels.cl ./

# Rule to build all files generated by this target.
CMakeFiles/MatrixMultiplicationCPPKernel.dir/build: bin/x86_64/Release/MatrixMultiplicationCPPKernel
.PHONY : CMakeFiles/MatrixMultiplicationCPPKernel.dir/build

CMakeFiles/MatrixMultiplicationCPPKernel.dir/requires: CMakeFiles/MatrixMultiplicationCPPKernel.dir/src/MatrixMultiplicationCPPKernel.cpp.o.requires
.PHONY : CMakeFiles/MatrixMultiplicationCPPKernel.dir/requires

CMakeFiles/MatrixMultiplicationCPPKernel.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/MatrixMultiplicationCPPKernel.dir/cmake_clean.cmake
.PHONY : CMakeFiles/MatrixMultiplicationCPPKernel.dir/clean

CMakeFiles/MatrixMultiplicationCPPKernel.dir/depend:
	cd /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl /media/dihong/2TB_RAID/MyWorks/DeepNN-Implementations/code/cl/CMakeFiles/MatrixMultiplicationCPPKernel.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/MatrixMultiplicationCPPKernel.dir/depend

