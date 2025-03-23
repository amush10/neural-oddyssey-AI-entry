# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.14)
project(mmdeploy-example)

if (NOT (${CMAKE_PROJECT_NAME} STREQUAL "MMDeploy"))
    find_package(MMDeploy REQUIRED)
endif ()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

function(add_example task folder name)
    if ((NOT task) OR (task IN_LIST MMDEPLOY_TASKS))
        # Search for c/cpp sources
        file(GLOB _SRCS ${folder}/${name}.c*)
        add_executable(${name} ${_SRCS})
        if (NOT (MSVC OR APPLE))
            # Disable new dtags so that executables can run even without LD_LIBRARY_PATH set
            target_link_libraries(${name} PRIVATE -Wl,--disable-new-dtags)
        endif ()
        if (MMDEPLOY_BUILD_SDK_MONOLITHIC)
            target_link_libraries(${name} PRIVATE mmdeploy ${OpenCV_LIBS})
        else ()
            # Load MMDeploy modules
            mmdeploy_load_static(${name} MMDeployStaticModules)
            mmdeploy_load_dynamic(${name} MMDeployDynamicModules)
            # Link to MMDeploy libraries
            target_link_libraries(${name} PRIVATE MMDeployLibs ${OpenCV_LIBS})
        endif ()
        install(TARGETS ${name} RUNTIME DESTINATION bin)
    endif ()
endfunction()

add_example(pose_detector src det_pose)
