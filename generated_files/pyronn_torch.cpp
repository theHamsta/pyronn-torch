#ifdef _MSC_BUILD
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>


void Cone_Backprojection3D_Kernel_Launcher(const float *sinogram_ptr, float *out, const float *projection_matrices, const int number_of_projections,
                                          const int volume_width, const int volume_height, const int volume_depth,
                                          const float volume_spacing_x, const float volume_spacing_y, const float volume_spacing_z,
                                          const float volume_origin_x, const float volume_origin_y, const float volume_origin_z,
                                          const int detector_width, const int detector_height, const float projection_multiplier);
    

void Cone_Projection_Kernel_Launcher(const float* volume_ptr, float *out, const float *inv_AR_matrix, const float *src_points, 
                                    const int number_of_projections, const int volume_width, const int volume_height, const int volume_depth, 
                                    const float volume_spacing_x, const float volume_spacing_y, const float volume_spacing_z,
                                    const int detector_width, const int detector_height, const float step_size);
    

void Cone_Projection_Kernel_Tex_Interp_Launcher(
    const float *volume_ptr, float *out,
    const float *inv_AR_matrix, const float *src_points,
    const int number_of_projections, const int volume_width,
    const int volume_height, const int volume_depth,
    const float volume_spacing_x, const float volume_spacing_y,
    const float volume_spacing_z, const int detector_width,
    const int detector_height, const float step_size);
    

void Parallel_Backprojection2D_Kernel_Launcher(const float *sinogram_ptr, float *out, const float *ray_vectors, const int number_of_projections,
                                               const int volume_width, const int volume_height, const float volume_spacing_x, const float volume_spacing_y,
                                               const float volume_origin_x, const float volume_origin_y,
                                               const int detector_size, const float detector_spacing, const float detector_origin);
    

void Parallel_Projection2D_Kernel_Launcher(
    const float *volume_ptr, float *out, const float *ray_vectors,
    const int number_of_projections, const int volume_width,
    const int volume_height, const float volume_spacing_x,
    const float volume_spacing_y, const float volume_origin_x,
    const float volume_origin_y, const int detector_size,
    const float detector_spacing, const float detector_origin);
    
using namespace pybind11::literals;





void call_Cone_Backprojection3D_Kernel_Launcher(at::Tensor& matrices, at::Tensor& projection, float projection_multiplier, at::Tensor& volume, float volume_origin_x, float volume_origin_y, float volume_origin_z, float volume_spacing_x, float volume_spacing_y, float volume_spacing_z)
{
   float * RESTRICT _data_volume = volume.data_ptr<float>();
   float * RESTRICT const _data_matrices = matrices.data_ptr<float>();
   float * RESTRICT const _data_projection = projection.data_ptr<float>();
   int64_t const _size_matrices_0 = matrices.size(0);
   int64_t const _size_projection_1 = projection.size(1);
   int64_t const _size_projection_2 = projection.size(2);
   int64_t const _size_volume_0 = volume.size(0);
   int64_t const _size_volume_1 = volume.size(1);
   int64_t const _size_volume_2 = volume.size(2);
   Cone_Backprojection3D_Kernel_Launcher(_data_projection, _data_volume, _data_matrices, _size_matrices_0, _size_volume_2, _size_volume_1, _size_volume_0, volume_spacing_x, volume_spacing_y, volume_spacing_z, volume_origin_x, volume_origin_y, volume_origin_z, _size_projection_2, _size_projection_1, projection_multiplier);
}

void call_Cone_Projection_Kernel_Launcher(at::Tensor& inv_matrices, at::Tensor& projection, at::Tensor& source_points, float step_size, at::Tensor& volume, float volume_spacing_x, float volume_spacing_y, float volume_spacing_z)
{
   float * RESTRICT _data_projection = projection.data_ptr<float>();
   float * RESTRICT const _data_inv_matrices = inv_matrices.data_ptr<float>();
   float * RESTRICT const _data_source_points = source_points.data_ptr<float>();
   float * RESTRICT const _data_volume = volume.data_ptr<float>();
   int64_t const _size_projection_1 = projection.size(1);
   int64_t const _size_projection_2 = projection.size(2);
   int64_t const _size_source_points_0 = source_points.size(0);
   int64_t const _size_volume_0 = volume.size(0);
   int64_t const _size_volume_1 = volume.size(1);
   int64_t const _size_volume_2 = volume.size(2);
   Cone_Projection_Kernel_Launcher(_data_volume, _data_projection, _data_inv_matrices, _data_source_points, _size_source_points_0, _size_volume_2, _size_volume_1, _size_volume_0, volume_spacing_x, volume_spacing_y, volume_spacing_z, _size_projection_2, _size_projection_1, step_size);
}

void call_Cone_Projection_Kernel_Tex_Interp_Launcher(at::Tensor& inv_matrices, at::Tensor& projection, at::Tensor& source_points, float step_size, at::Tensor& volume, float volume_spacing_x, float volume_spacing_y, float volume_spacing_z)
{
   float * RESTRICT _data_projection = projection.data_ptr<float>();
   float * RESTRICT const _data_inv_matrices = inv_matrices.data_ptr<float>();
   float * RESTRICT const _data_source_points = source_points.data_ptr<float>();
   float * RESTRICT const _data_volume = volume.data_ptr<float>();
   int64_t const _size_projection_1 = projection.size(1);
   int64_t const _size_projection_2 = projection.size(2);
   int64_t const _size_source_points_0 = source_points.size(0);
   int64_t const _size_volume_0 = volume.size(0);
   int64_t const _size_volume_1 = volume.size(1);
   int64_t const _size_volume_2 = volume.size(2);
   Cone_Projection_Kernel_Tex_Interp_Launcher(_data_volume, _data_projection, _data_inv_matrices, _data_source_points, _size_source_points_0, _size_volume_2, _size_volume_1, _size_volume_0, volume_spacing_x, volume_spacing_y, volume_spacing_z, _size_projection_2, _size_projection_1, step_size);
}

void call_Parallel_Projection2D_Kernel_Launcher(float detector_origin, float detector_spacing, at::Tensor& projections_1d, at::Tensor& ray_vectors, float volume_origin_x, float volume_origin_y, at::Tensor& volume_slice, float volume_spacing_x, float volume_spacing_y)
{
   float * RESTRICT _data_projections_1d = projections_1d.data_ptr<float>();
   float * RESTRICT const _data_ray_vectors = ray_vectors.data_ptr<float>();
   float * RESTRICT const _data_volume_slice = volume_slice.data_ptr<float>();
   int64_t const _size_projections_1d_1 = projections_1d.size(1);
   int64_t const _size_ray_vectors_0 = ray_vectors.size(0);
   int64_t const _size_volume_slice_0 = volume_slice.size(0);
   int64_t const _size_volume_slice_1 = volume_slice.size(1);
   Parallel_Projection2D_Kernel_Launcher(_data_volume_slice, _data_projections_1d, _data_ray_vectors, _size_ray_vectors_0, _size_volume_slice_1, _size_volume_slice_0, volume_spacing_x, volume_spacing_y, volume_origin_x, volume_origin_y, _size_projections_1d_1, detector_spacing, detector_origin);
}

void call_Parallel_Backprojection2D_Kernel_Launcher(float detector_origin, float detector_spacing, at::Tensor& projections_1d, at::Tensor& ray_vectors, float volume_origin_x, float volume_origin_y, at::Tensor& volume_slice, float volume_spacing_x, float volume_spacing_y)
{
   float * RESTRICT _data_volume_slice = volume_slice.data_ptr<float>();
   float * RESTRICT const _data_projections_1d = projections_1d.data_ptr<float>();
   float * RESTRICT const _data_ray_vectors = ray_vectors.data_ptr<float>();
   int64_t const _size_projections_1d_1 = projections_1d.size(1);
   int64_t const _size_ray_vectors_0 = ray_vectors.size(0);
   int64_t const _size_volume_slice_0 = volume_slice.size(0);
   int64_t const _size_volume_slice_1 = volume_slice.size(1);
   Parallel_Backprojection2D_Kernel_Launcher(_data_projections_1d, _data_volume_slice, _data_ray_vectors, _size_ray_vectors_0, _size_volume_slice_1, _size_volume_slice_0, volume_spacing_x, volume_spacing_y, volume_origin_x, volume_origin_y, _size_projections_1d_1, detector_spacing, detector_origin);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
   m.def("call_Cone_Backprojection3D_Kernel_Launcher", &call_Cone_Backprojection3D_Kernel_Launcher, "matrices"_a, "projection"_a, "projection_multiplier"_a, "volume"_a, "volume_origin_x"_a, "volume_origin_y"_a, "volume_origin_z"_a, "volume_spacing_x"_a, "volume_spacing_y"_a, "volume_spacing_z"_a);
   m.def("call_Cone_Projection_Kernel_Launcher", &call_Cone_Projection_Kernel_Launcher, "inv_matrices"_a, "projection"_a, "source_points"_a, "step_size"_a, "volume"_a, "volume_spacing_x"_a, "volume_spacing_y"_a, "volume_spacing_z"_a);
   m.def("call_Cone_Projection_Kernel_Tex_Interp_Launcher", &call_Cone_Projection_Kernel_Tex_Interp_Launcher, "inv_matrices"_a, "projection"_a, "source_points"_a, "step_size"_a, "volume"_a, "volume_spacing_x"_a, "volume_spacing_y"_a, "volume_spacing_z"_a);
   m.def("call_Parallel_Projection2D_Kernel_Launcher", &call_Parallel_Projection2D_Kernel_Launcher, "detector_origin"_a, "detector_spacing"_a, "projections_1d"_a, "ray_vectors"_a, "volume_origin_x"_a, "volume_origin_y"_a, "volume_slice"_a, "volume_spacing_x"_a, "volume_spacing_y"_a);
   m.def("call_Parallel_Backprojection2D_Kernel_Launcher", &call_Parallel_Backprojection2D_Kernel_Launcher, "detector_origin"_a, "detector_spacing"_a, "projections_1d"_a, "ray_vectors"_a, "volume_origin_x"_a, "volume_origin_y"_a, "volume_slice"_a, "volume_spacing_x"_a, "volume_spacing_y"_a);
}