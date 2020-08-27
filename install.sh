python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

echo "compiling rasterizer"
TF_INC=./env/lib/python3.7/site-packages/tensorflow_core/include
TF_LIB=./env/lib/python3.7/site-packages/tensorflow_core
# you might need the following to successfully compile the third-party library
#ln -s ./env/lib/python3.7/site-packages/tensorflow_core/libtensorflow_framework.so.1 ./env/lib/python3.7/site-packages/tensorflow_core/libtensorflow_framework.so
mkdir ./tools/kernels
g++ -std=c++11 \
    -shared ./tools/src_mesh_renderer/rasterize_triangles_grad.cc ./tools/src_mesh_renderer/rasterize_triangles_op.cc ./tools/src_mesh_renderer/rasterize_triangles_impl.cc ./tools/src_mesh_renderer/rasterize_triangles_impl.h \
    -o ./tools/kernels/rasterize_triangles_kernel.so -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 \
    -I$TF_INC -L$TF_LIB -ltensorflow_framework -O2
