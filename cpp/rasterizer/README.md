# rasterizer (C++/CUDA)

Custom Gaussian Splatting rasterization kernel, bound into Python via
PyTorch's custom C++/CUDA extension mechanism (torch.utils.cpp_extension).

Plan:
    1. Study a permissively-licensed reference rasterizer implementation
       in depth (tile-based rasterization, per-tile depth sorting,
       alpha compositing, backward pass for gradients w.r.t. Gaussian
       parameters).
    2. Re-implement the forward pass yourself to build real understanding,
       validating numerically against the reference.
    3. Make a meaningful, honestly-attributable extension -- e.g.,
       support for a non-pinhole (fisheye) camera model, or a memory
       layout change for better tile-sort performance on long/thin
       highway scenes -- so the project can honestly claim understanding
       and extension, not reproduction.

This is the single highest-leverage piece of the repo for demonstrating
production-quality native-language systems work.
