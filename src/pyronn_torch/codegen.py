#
# Copyright © 2020 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""


import argparse
import tempfile
from glob import glob
from os import makedirs
from os.path import basename, dirname, join
from shutil import copyfile, copytree, rmtree

try:
    import pystencils
    from pystencils.cpu.cpujit import get_cache_config
    from pystencils.data_types import TypedSymbol, create_type
    from pystencils.kernelparameters import FieldPointerSymbol, FieldShapeSymbol
    from pystencils_autodiff.backends.astnodes import TorchModule
    from pystencils_autodiff.framework_integration.astnodes import \
        CustomFunctionCall
    from pystencils_autodiff.framework_integration.printer import \
        FrameworkIntegrationPrinter

    volume = pystencils.fields('volume: float32[3d]')
    projection = pystencils.fields('projection: float32[3d]')
    projection_matrices = pystencils.fields('matrices: float32[3d]')
    inv_matrices = pystencils.fields('inv_matrices: float32[3d]')
    source_points = pystencils.fields('source_points: float32[1d]')
    volume_slice = pystencils.fields('volume_slice: float32[2d]')
    projections_1d = pystencils.fields('projections_1d: float32[2d]')
    ray_vectors = pystencils.fields('ray_vectors: float32[2d]')

    FUNCTIONS = {
            'Cone_Backprojection3D_Kernel_Launcher': CustomFunctionCall('Cone_Backprojection3D_Kernel_Launcher',
                                                 FieldPointerSymbol(projection.name, projection.dtype, const=True),
                                                 FieldPointerSymbol(volume.name, volume.dtype, const=False),
                                                 FieldPointerSymbol(projection_matrices.name,
                                                                    projection_matrices.dtype, const=True),
                                                 FieldShapeSymbol(['matrices'], 0),
                                                 *[FieldShapeSymbol(['volume'], i) for i in range(2, -1, -1)],
                                                 TypedSymbol('volume_spacing_x', create_type('float32'), const=True),
                                                 TypedSymbol('volume_spacing_y', create_type('float32'), const=True),
                                                 TypedSymbol('volume_spacing_z', create_type('float32'), const=True),
                                                 TypedSymbol('volume_origin_x', create_type('float32'), const=True),
                                                 TypedSymbol('volume_origin_y', create_type('float32'), const=True),
                                                 TypedSymbol('volume_origin_z', create_type('float32'), const=True),
                                                 *[FieldShapeSymbol(['projection'], i) for i in range(2, 0, -1)],
                                                 TypedSymbol('projection_multiplier',
                                                             create_type('float32'), const=True), backend='gpucuda',
                                                 fields_accessed=[volume, projection, projection_matrices], custom_signature="""
void Cone_Backprojection3D_Kernel_Launcher(const float *sinogram_ptr, float *out, const float *projection_matrices, const int number_of_projections,
                                          const int volume_width, const int volume_height, const int volume_depth,
                                          const float volume_spacing_x, const float volume_spacing_y, const float volume_spacing_z,
                                          const float volume_origin_x, const float volume_origin_y, const float volume_origin_z,
                                          const int detector_width, const int detector_height, const float projection_multiplier);
    """),  # noqa
    'Cone_Projection_Kernel_Launcher': CustomFunctionCall('Cone_Projection_Kernel_Launcher',
                                                 FieldPointerSymbol(volume.name, volume.dtype, const=True),
                                                 FieldPointerSymbol(projection.name, projection.dtype, const=False),
                                                 FieldPointerSymbol(inv_matrices.name,
                                                                    inv_matrices.dtype, const=True),
                                                 FieldPointerSymbol(source_points.name,
                                                                    source_points.dtype, const=True),
                                                 FieldShapeSymbol([source_points.name], 0),
                                                 *[FieldShapeSymbol(['volume'], i) for i in range(2, -1, -1)],
                                                 TypedSymbol('volume_spacing_x', create_type('float32'), const=True),
                                                 TypedSymbol('volume_spacing_y', create_type('float32'), const=True),
                                                 TypedSymbol('volume_spacing_z', create_type('float32'), const=True),
                                                 *[FieldShapeSymbol(['projection'], i) for i in range(2, 0, -1)],
                                                 TypedSymbol('step_size', create_type('float32'), const=True),
                                                 backend='gpucuda',
                                                 fields_accessed=[volume, projection, inv_matrices, source_points],
                                                 custom_signature="""
void Cone_Projection_Kernel_Launcher(const float* volume_ptr, float *out, const float *inv_AR_matrix, const float *src_points, 
                                    const int number_of_projections, const int volume_width, const int volume_height, const int volume_depth, 
                                    const float volume_spacing_x, const float volume_spacing_y, const float volume_spacing_z,
                                    const int detector_width, const int detector_height, const float step_size);
    """),  # noqa
    'Cone_Projection_Kernel_Tex_Interp_Launcher': CustomFunctionCall('Cone_Projection_Kernel_Tex_Interp_Launcher',
                                                 FieldPointerSymbol(volume.name, volume.dtype, const=True),
                                                 FieldPointerSymbol(projection.name, projection.dtype, const=False),
                                                 FieldPointerSymbol(inv_matrices.name,
                                                                    inv_matrices.dtype, const=True),
                                                 FieldPointerSymbol(source_points.name,
                                                                    source_points.dtype, const=True),
                                                 FieldShapeSymbol([source_points.name], 0),
                                                 *[FieldShapeSymbol(['volume'], i) for i in range(2, -1, -1)],
                                                 TypedSymbol('volume_spacing_x', create_type('float32'), const=True),
                                                 TypedSymbol('volume_spacing_y', create_type('float32'), const=True),
                                                 TypedSymbol('volume_spacing_z', create_type('float32'), const=True),
                                                 *[FieldShapeSymbol(['projection'], i) for i in range(2, 0, -1)],
                                                 TypedSymbol('step_size', create_type('float32'), const=True),
                                                 backend='gpucuda',
                                                 fields_accessed=[volume, projection, inv_matrices, source_points],
                                                 custom_signature="""
void Cone_Projection_Kernel_Tex_Interp_Launcher(
    const float *volume_ptr, float *out,
    const float *inv_AR_matrix, const float *src_points,
    const int number_of_projections, const int volume_width,
    const int volume_height, const int volume_depth,
    const float volume_spacing_x, const float volume_spacing_y,
    const float volume_spacing_z, const int detector_width,
    const int detector_height, const float step_size);
    """),  # noqa
    'Parallel_Projection2D_Kernel_Launcher': CustomFunctionCall('Parallel_Projection2D_Kernel_Launcher',
                                                 FieldPointerSymbol(volume_slice.name, volume_slice.dtype, const=True),
                                                 FieldPointerSymbol(projections_1d.name,
                                                                    projections_1d.dtype, const=False),
                                                 FieldPointerSymbol(ray_vectors.name,
                                                                    ray_vectors.dtype, const=True),
                                                 FieldShapeSymbol([ray_vectors.name], 0),
                                                 *[FieldShapeSymbol(['volume_slice'], i) for i in range(1, -1, -1)],
                                                 TypedSymbol('volume_spacing_x', create_type('float32'), const=True),
                                                 TypedSymbol('volume_spacing_y', create_type('float32'), const=True),
                                                 TypedSymbol('volume_origin_x', create_type('float32'), const=True),
                                                 TypedSymbol('volume_origin_y', create_type('float32'), const=True),
                                                 FieldShapeSymbol([projections_1d.name], 1),
                                                 TypedSymbol('detector_spacing', create_type('float32'), const=True),
                                                 TypedSymbol('detector_origin', create_type('float32'), const=True),
                                                 backend='gpucuda',
                                                 fields_accessed=[volume_slice, projections_1d, ray_vectors], custom_signature="""
void Parallel_Projection2D_Kernel_Launcher(
    const float *volume_ptr, float *out, const float *ray_vectors,
    const int number_of_projections, const int volume_width,
    const int volume_height, const float volume_spacing_x,
    const float volume_spacing_y, const float volume_origin_x,
    const float volume_origin_y, const int detector_size,
    const float detector_spacing, const float detector_origin);
    """),  # noqa
    'Parallel_Backprojection2D_Kernel_Launcher': CustomFunctionCall('Parallel_Backprojection2D_Kernel_Launcher',
                                                 FieldPointerSymbol(projections_1d.name,
                                                                    projections_1d.dtype, const=True),
                                                 FieldPointerSymbol(volume_slice.name, volume_slice.dtype, const=False),
                                                 FieldPointerSymbol(ray_vectors.name,
                                                                    ray_vectors.dtype, const=True),
                                                 FieldShapeSymbol([ray_vectors.name], 0),
                                                 *[FieldShapeSymbol(['volume_slice'], i) for i in range(1, -1, -1)],
                                                 TypedSymbol('volume_spacing_x', create_type('float32'), const=True),
                                                 TypedSymbol('volume_spacing_y', create_type('float32'), const=True),
                                                 TypedSymbol('volume_origin_x', create_type('float32'), const=True),
                                                 TypedSymbol('volume_origin_y', create_type('float32'), const=True),
                                                 FieldShapeSymbol([projections_1d.name], 1),
                                                 TypedSymbol('detector_spacing', create_type('float32'), const=True),
                                                 TypedSymbol('detector_origin', create_type('float32'), const=True),
                                                 backend='gpucuda',
                                                 fields_accessed=[volume_slice, projections_1d, ray_vectors], custom_signature="""
void Parallel_Backprojection2D_Kernel_Launcher(const float *sinogram_ptr, float *out, const float *ray_vectors, const int number_of_projections,
                                               const int volume_width, const int volume_height, const float volume_spacing_x, const float volume_spacing_y,
                                               const float volume_origin_x, const float volume_origin_y,
                                               const int detector_size, const float detector_spacing, const float detector_origin);
    """),  # noqa
}
except Exception as e:
    import warnings
    warnings.warn(str(e))
    TorchModule = None


def get_pyronn_cuda_kernels():
    return glob(join(dirname(__file__), 'PYRO-NN-Layers', '*.cu.cc'))


def get_pyronn_include_paths():
    return join(dirname(__file__), 'PYRO-NN-Layers')


def generate_shared_object(output_folder=None,
                           source_files=None,
                           show_code=False,
                           framework_module_class=TorchModule,
                           generate_code_only=False,
                           update_repo_files=False):

    object_cache = get_cache_config()['object_cache']

    module_name = 'pyronn_torch_cpp'

    if not output_folder:
        output_folder = join(dirname(__file__), '..', '..', 'generated_files')

    if not source_files:
        source_files = glob(join(dirname(__file__), 'PYRO-NN-Layers', '*.cu.cc'))

    cuda_sources = []
    makedirs(join(object_cache, module_name), exist_ok=True)
    rmtree(join(object_cache, module_name, 'helper_headers'), ignore_errors=True)
    copytree(join(dirname(__file__), 'PYRO-NN-Layers', 'helper_headers'),
             join(object_cache, module_name, 'helper_headers'))
    if update_repo_files:
        rmtree(join(output_folder, 'helper_headers'), ignore_errors=True)
        copytree(join(dirname(__file__), 'PYRO-NN-Layers', 'helper_headers'),
                 join(output_folder, 'helper_headers'))

    for s in source_files:
        dst = join(object_cache, module_name, basename(s).replace('.cu.cc', '.cu'))
        copyfile(s, dst)  # Torch only accepts *.cu as CUDA
        cuda_sources.append(dst)
        if update_repo_files:
            dst = join(output_folder, basename(s).replace('.cu.cc', '.cu'))
            copyfile(s, dst)  # Torch only accepts *.cu as CUDA

    module = framework_module_class(module_name, FUNCTIONS.values())

    if show_code:
        pystencils.show_code(module, custom_backend=FrameworkIntegrationPrinter())

    if generate_code_only:
        return module

    extension = module.compile(extra_source_files=cuda_sources,
                               extra_cuda_flags=['-arch=sm_35'],
                               with_cuda=True,
                               compile_module_name=module_name)

    shared_object_file = module.compiled_file
    copyfile(shared_object_file, join(output_folder, module_name + '.so'))

    if update_repo_files:
        with open(join(output_folder, 'pyronn_torch.cpp'), 'w') as f:
            f.write(pystencils.get_code_str(module, custom_backend=FrameworkIntegrationPrinter()))

    return extension


def compile_shared_object(output_folder=None, source_files=None):
    """
    Same as generate_shared_object without pystencils dependency
    """
    object_cache = tempfile.gettempdir()
    module_name = 'pyronn_torch_cpp'

    if not output_folder:
        output_folder = dirname(__file__)

    if not source_files:
        source_files = get_pyronn_cuda_kernels()

    generated_file = join(dirname(__file__), 'pyronn_torch.cpp')

    cuda_sources = []
    makedirs(join(object_cache, module_name), exist_ok=True)
    rmtree(join(object_cache, module_name, 'helper_headers'), ignore_errors=True)
    copytree(join(dirname(__file__), 'PYRO-NN-Layers', 'helper_headers'),
             join(object_cache, module_name, 'helper_headers'))

    for s in source_files:
        dst = join(object_cache, module_name, basename(s).replace('.cu.cc', '.cu'))
        copyfile(s, dst)  # Torch only accepts *.cu as CUDA
        cuda_sources.append(dst)

    # extension = module.compile(extra_source_files=cuda_sources,
        # extra_cuda_flags=['-arch=sm_35'],
        # with_cuda=True,
        # compile_module_name=module_name)

    from torch.utils.cpp_extension import load
    torch_extension = load(module_name,
                           [generated_file] + cuda_sources,
                           with_cuda=True,
                           extra_cflags=[],
                           extra_cuda_cflags=['-arch=sm_35'],
                           build_directory=join(object_cache, module_name),
                           extra_include_paths=[])
    shared_object_file = join(object_cache, module_name, module_name + '.so')
    copyfile(shared_object_file, join(output_folder, module_name + '.so'))
    return torch_extension


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-folder',  default=None)
    parser.add_argument('--source-files',  default=None)
    args = parser.parse_args()

    generate_shared_object(args.output_folder, args.source_files, show_code=True, update_repo_files=True)
    # compile_shared_object(args.output_folder, args.source_files)


if __name__ == '__main__':
    main()
