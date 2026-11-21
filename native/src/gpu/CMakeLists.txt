# The dispatcher is always built, so every build has the same symbols; a backend is added only if CMake
# found its toolkit (see CTA_GPU in native/CMakeLists.txt).
target_sources(_native PRIVATE dispatch.cpp)

if(CTA_HAVE_METAL)
  target_sources(_native PRIVATE metal_solver.mm)
  set_source_files_properties(metal_solver.mm TARGET_DIRECTORY _native PROPERTIES
                              COMPILE_OPTIONS "-fobjc-arc;-ffp-contract=off")
  target_compile_definitions(_native PRIVATE CTA_HAVE_METAL=1)
  target_link_libraries(_native PRIVATE ${CTA_METAL_FRAMEWORK} ${CTA_FOUNDATION_FRAMEWORK})
endif()

if(CTA_HAVE_CUDA)
  target_sources(_native PRIVATE cuda_solver.cu)
  # No fused multiply-add: the solver's multiply-then-add must round like the CPU's.
  set_source_files_properties(cuda_solver.cu TARGET_DIRECTORY _native PROPERTIES COMPILE_OPTIONS "-fmad=false")
  target_compile_definitions(_native PRIVATE CTA_HAVE_CUDA=1)
  target_link_libraries(_native PRIVATE CUDA::cudart_static)
endif()
