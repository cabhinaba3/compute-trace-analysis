# One file per trace format, plus the 3D-parallel training graphs.
target_sources(_native PRIVATE
  chakra.cpp
  goal.cpp
  jtop.cpp
  kineto.cpp
  ncu.cpp
  nsys.cpp
  training.cpp
  trtexec.cpp)
