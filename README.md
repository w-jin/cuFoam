## cuFoam

`cuFoam`(cuda based linear equations solver for OpenFOAM) 是一个为OpenFOAM编写的线性方程组求解器。在OpenFOAM中，求解微分方程需要求解稀疏线性方程组，稀疏线性方程组一般通过迭代方法求解。本项目基于nvidia的cuda计算平台实现了常见的线性方程组迭代求解方法，并将之集成到OpenFOAM中，为OpenFOAM的计算过程加速。

## 构建

构建之前先加载OpenFOAM环境变量，例如：

```
source /opt/OpenFOAM/OpenFOAM-v1912/etc/bashrc
```

根据自己的安装位置和OpenFOAM版本进行调整。

CUDA和OpenFOAM两部分的代码必须分开编译。编译CUDA相关代码时使用nvcc编译器，编译OpenFOAM时使用标准c++编译器，不能处理GPU代码，但可以链接到nvcc生成的库文件。首先进入gis目录，编译CUDA相关的代码，可以由以下命令完成：

```
cufoam/gis$ nvcc -Xcompiler -fPIC  -shared gis/gis.cu -o $FOAM_USER_LIBBIN/libGIS.so -lcudart
```

编译好的库文件将放在用户的OpenFOAM库目录下，如果此目录不存在，可以通过下面的命令创建：

```
$ mkdir -p $FOAM_USER_LIBBIN
```

也可以选择使用cmake进行编译：

```
cufoam/gis$ mkdir build
cufoam/gis$ cd build
cufoam/gis/build$ cmake ..
cufoam/gis/build$ make
cufoam/gis/build$ make install
```

同样地，cmake会把编译好的库文件libGIS.so安装到$FOAM_USER_LIBBIN下。

接下来编译OpenFOAM相关的部分。回到项目根目录，即有Make文件夹的目录，执行下面的命令：

```
cufoam$ wmake
```

生成的库libcuFoam.so将被安装到$FOAM_USER_LIBBIN下。

## 使用

正确构建后，需要的库文件已经被安装到$FOAM_USER_LIBBIN下，可以被OpenFOAM检索到。使用时在system/controlDict中添加一行：

```
libs ("libcuFoam.so");
```

注意libs和前括号间必须有空白分隔。然后修改system/fvSolution中的solver字段，如：

```
solvers
{
    p
    {
        solver          cuFoamCG;
        tolerance       1e-06;
        relTol          0.05;
    }
    // ...
```

cuFoamCG可以替换为cuFoamBiCG、cuFoamJacobi等方法，只要迭代方法可以收敛。

## 开发路线

目前已经实现的方法数量有限，以后会添加更多的求解方法，此外，还会考虑加入一些常用的预处理方法，以加快方法的收敛速度。

