from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


setup(
    name=f"uvm_pytorch",
    description="Unified Memory Tensors",
    keywords="unified_memory",
    version="0.5.28.2024",
    url="https://github.com/lessw2020/UVM_Tensor",
    packages=find_packages(),
    cmdclass={'build_ext': BuildExtension},
    ext_modules=[
        CUDAExtension(
            name='uvm_pytorch',
            sources=['managed_memory.cpp'],
            extra_compile_args={'cxx': [], 'nvcc': ['-lineinfo']}
        ),

    ],
)
