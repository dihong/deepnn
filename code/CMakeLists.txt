#################################################################################
# Copyright ©2014 Advanced Micro Devices, Inc. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# •	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# •	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
#  other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#################################################################################

set( EXECUTABLE_NAME proc )
set( SOURCE_FILES src/main.cpp src/setup.cpp src/common.cpp src/full.cpp src/full.forward.cpp src/full.dEdX.cpp src/full.dEdW.cpp src/full.dEdB.cpp src/softmax.cpp src/conv.cpp src/schedule.cpp
src/graph.cpp src/input.cpp src/SOIL/SOIL.c src/SOIL/SOIL.c src/SOIL/stb_image_aug.c src/SOIL/image_helper.c src/SOIL/image_DXT.c)
#set( EXTRA_FILES src/mat_mul.cl )

############################################################################

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.8.0 )
project( ${EXECUTABLE_NAME} )

if(CMAKE_BUILD_TYPE MATCHES "[Tt][Bb][Bb]")
	return( )
endif()

# Auto-select bitness based on platform
if( NOT BITNESS )
    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(BITNESS 64)
    else()
        set(BITNESS 32)
    endif()
endif()

# Select bitness for non-msvc platform. Can be specified as -DBITNESS=32/64 at command-line
if( NOT MSVC )
    set(BITNESS ${BITNESS} CACHE STRING "Specify bitness")
    set_property(CACHE BITNESS PROPERTY STRINGS "64" "32")
endif()

# Unset OPENCL_LIBRARIES, so that corresponding arch specific libs are found when bitness is changed
unset(OPENCL_LIBRARIES CACHE)

if( BITNESS EQUAL 64 )
    set(BITNESS_SUFFIX x86_64)
elseif( BITNESS EQUAL 32 )
    set(BITNESS_SUFFIX x86)
else()
    message( FATAL_ERROR "Bitness specified is invalid" )
endif()

# Set CMAKE_BUILD_TYPE (default = Release)
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
	set(CMAKE_BUILD_TYPE Release)
endif()

# Set platform
if( NOT UNIX )
	set(PLATFORM win)
else()
	set(PLATFORM lnx)
endif()

############################################################################

if(TARGET_PLATFORM EQUAL 32 OR TARGET_PLATFORM EQUAL 64)
    set(TARGET_PLATFORM ${TARGET_PLATFORM} CACHE STRING "Target platform type (32-bit or 64-bit)" FORCE)
else()
    if(CMAKE_SIZEOF_VOID_P MATCHES 8)
        set(TARGET_PLATFORM "64" CACHE STRING "Target platform type (32-bit or 64-bit)" FORCE)
    else()
        set(TARGET_PLATFORM "32" CACHE STRING "Target platform type (32-bit or 64-bit)" FORCE)
    endif()
endif()
message(STATUS "Target platform: ${TARGET_PLATFORM}-bit")


# Find OpenCL include
find_path( OPENCL_INCLUDE_DIRS 
    NAMES CL/cl.h
    HINTS include/AMDAPPSDK/
)
mark_as_advanced(OPENCL_INCLUDE_DIRS)


# Find SOIL include
find_path( SOIL_INCLUDE_DIRS 
    NAMES SOIL.h
    HINTS include/SOIL/
)
mark_as_advanced(SOIL_INCLUDE_DIRS)


# Find clFFT include
find_path( FFT_INCLUDE_DIRS 
    NAMES clFFT.h
    HINTS include/clFFT/
)
mark_as_advanced(FFT_INCLUDE_DIRS)


# Find private include
find_path( PRIVATE_INCLUDE_DIRS 
    NAMES main.hpp
    HINTS include/
)
mark_as_advanced(PRIVATE_INCLUDE_DIRS)

# Find OpenCL lib
find_library( OPENCL_LIBRARIES
	NAMES OpenCL
	HINTS lib/AMDAPPSDK
	PATH_SUFFIXES ${PLATFORM}${BITNESS} ${BITNESS_SUFFIX}
)
mark_as_advanced( OPENCL_LIBRARIES )

# Find FFT lib
find_library( FFT_LIBRARIES
	NAMES clFFT
	HINTS lib/clFFT
	PATH_SUFFIXES ${PLATFORM}${BITNESS} ${BITNESS_SUFFIX}
)
mark_as_advanced( FFT_LIBRARIES )


if( OPENCL_INCLUDE_DIRS STREQUAL "" OR OPENCL_LIBRARIES STREQUAL "")
	message( FATAL_ERROR "Could not locate OpenCL include & libs" )
endif( )

if( FFT_INCLUDE_DIRS STREQUAL "" OR FFT_LIBRARIES STREQUAL "")
	message( FATAL_ERROR "Could not locate clFFT include & libs" )
endif( )

############################################################################
# Tweaks for cygwin makefile to work with windows-style path

if( CYGWIN )
    set( PATHS_TO_CONVERT
           OPENCL_INCLUDE_DIRS
           OPENCL_LIBRARIES
           FFT_LIBRARIES
           PRIVATE_INCLUDE_DIRS
           SOIL_INCLUDE_DIRS
           FFT_INCLUDE_DIRS
       )
       
    foreach( pathVar ${PATHS_TO_CONVERT} )
        # Convert windows paths to cyg linux absolute path
        execute_process( COMMAND cygpath -ua ${${pathVar}}
                            OUTPUT_VARIABLE ${pathVar}
                            OUTPUT_STRIP_TRAILING_WHITESPACE
                       )
    endforeach( pathVar )
endif( )
############################################################################

set( COMPILER_FLAGS "" )
set( LINKER_FLAGS "" )
set( ADDITIONAL_LIBRARIES "GL" )




#######END-OF-CIMG#########

file(GLOB INCLUDE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h" )
include_directories( ${OPENCL_INCLUDE_DIRS} ${PRIVATE_INCLUDE_DIRS} ${SOIL_INCLUDE_DIRS} ${FFT_INCLUDE_DIRS})

add_executable( ${EXECUTABLE_NAME} ${SOURCE_FILES} ${INCLUDE_FILES} ${EXTRA_FILES})


# gcc/g++ specific compile options
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    set( COMPILER_FLAGS "${COMPILER_FLAGS} -msse2 -std=c++11" )
    
    # Note: "rt" is not present on mingw
    if( UNIX )
		if( CMAKE_BUILD_TYPE STREQUAL "Debug" )
			set( COMPILER_FLAGS " -g " )
		endif( )
        set( ADDITIONAL_LIBRARIES ${ADDITIONAL_LIBRARIES} "rt" "pthread")
    endif( )
    
    if( BITNESS EQUAL 32 )
        set( COMPILER_FLAGS "${COMPILER_FLAGS} -m32 " )
        set( LINKER_FLAGS "${LINKER_FLAGS} -m32 " )
    else( )
        set( COMPILER_FLAGS "${COMPILER_FLAGS} -m64 " )
        set( LINKER_FLAGS "${LINKER_FLAGS} -m64 " )
    endif( )
    
    set( COMPILER_FLAGS "${COMPILER_FLAGS} ${EXTRA_COMPILER_FLAGS_GXX} " )
    set( LINKER_FLAGS "${LINKER_FLAGS} ${EXTRA_LINKER_FLAGS_GXX} " )
    set( ADDITIONAL_LIBRARIES ${ADDITIONAL_LIBRARIES} ${EXTRA_LIBRARIES_GXX} )
elseif( MSVC )
    # Samples can specify additional libs/flags using EXTRA* defines
	add_definitions( "/W3 /D_CRT_SECURE_NO_WARNINGS /wd4005 /wd4996 /nologo" )

    set( COMPILER_FLAGS "${COMPILER_FLAGS} ${EXTRA_COMPILER_FLAGS_MSVC} " )
    set( LINKER_FLAGS "${LINKER_FLAGS} ${EXTRA_LINKER_FLAGS_MSVC}  /SAFESEH:NO ")
    set( ADDITIONAL_LIBRARIES ${ADDITIONAL_LIBRARIES} ${EXTRA_LIBRARIES_MSVC} )
endif( )


set_target_properties( ${EXECUTABLE_NAME} PROPERTIES
                        COMPILE_FLAGS ${COMPILER_FLAGS}
                        LINK_FLAGS ${LINKER_FLAGS}
                     )
target_link_libraries( ${EXECUTABLE_NAME} ${OPENCL_LIBRARIES} ${FFT_LIBRARIES} ${ADDITIONAL_LIBRARIES} )

# Set output directory to bin
if( MSVC )
	set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin/${BITNESS_SUFFIX})
else()
	set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin/${BITNESS_SUFFIX}/${CMAKE_BUILD_TYPE})
endif()

# Copy extra files to binary directory
foreach( extra_file ${EXTRA_FILES} )
    add_custom_command(
        TARGET ${EXECUTABLE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/${extra_file}  ${EXECUTABLE_OUTPUT_PATH}/${CMAKE_CFG_INTDIR}
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/${extra_file}  ./
        )
endforeach( extra_file )

# Group sample based on FOLDER_GROUP defined in parent folder
if( FOLDER_GROUP )
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES FOLDER ${FOLDER_GROUP})
endif( )
