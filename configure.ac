#
# Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See file LICENSE for terms.
#
AC_PREREQ([2.63])

AC_COPYRIGHT([Copyright (c) 2019, NVIDIA CORPORATION & AFFILIATES. All rights reserved.])

define([nccl_rdma_sharp_plugins_ver_major], 2)
define([nccl_rdma_sharp_plugins_ver_minor], 7)

AC_INIT([nccl-rdma-sharp-plugins], [nccl_rdma_sharp_plugins_ver_major.nccl_rdma_sharp_plugins_ver_minor], [support@mellanox.com], [],[http://github.com/Mellanox/nccl-rdma-sharp-plugins])

AM_INIT_AUTOMAKE([1.10 foreign tar-ustar subdir-objects])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])
AM_MAINTAINER_MODE
AC_CONFIG_MACRO_DIR([m4])

AC_USE_SYSTEM_EXTENSIONS
AC_GNU_SOURCE
AC_CONFIG_HEADERS([config.h])

RPM_RELEASE=1
MAJOR_VERSION=nccl_rdma_sharp_plugins_ver_major
MINOR_VERSION=nccl_rdma_sharp_plugins_ver_minor
VERSION=$MAJOR_VERSION.$MINOR_VERSION

AC_SUBST(RPM_RELEASE)
AC_SUBST(VERSION)
AC_SUBST(MAJOR_VERSION)
AC_SUBST(MINOR_VERSION)
AC_SUBST([BUILD_DATE], [$(date +'%b/%d/%Y')])
AC_SUBST([BUILD_TIME], [$(date +'%H:%M:%S')])

# Checks for programs.
AC_GNU_SOURCE
AC_PROG_CC
AC_PROG_CC_STDC
AC_PROG_CXX
AM_PROG_AS
AC_PROG_SED
AC_PROG_INSTALL
AC_PROG_LIBTOOL
AC_HEADER_STDC
LT_LIB_M

AC_ARG_ENABLE([debug],AS_HELP_STRING([--enable-debug], [Enable extra debugging code (default is NO).]),
              [], [enable_debug=no])

if test $enable_debug = yes; then
	AC_DEFINE([ENABLE_DEBUG], [1], [Enable debugging code])
	CFLAGS="$CFLAGS -O0 -g3 -Wall -Werror"
else
	CFLAGS="$CFLAGS -O3 -DNDEBUG -Wall -Werror"
fi

#check for cuda
AC_ARG_WITH([cuda],
            [AC_HELP_STRING([--with-cuda=PATH],
            [Path to non-standard CUDA installation])],
            [AS_IF([test -d $withval/lib64], [cuda_libdir="lib64"], [cuda_libdir="lib"])
             CFLAGS="-I$withval/include $CFLAGS"
             LDFLAGS="-L$withval/$cuda_libdir $LDFLAGS"],
            [CFLAGS="-I/usr/local/cuda/include $CFLAGS"
             LDFLAGS="-L/usr/local/cuda/$cuda_libdir $LDFLAGS"])

AC_CHECK_HEADER(       [cuda_runtime.h], [], [AC_MSG_FAILURE([CUDA runtime header files not found])])
AC_CHECK_LIB([cudart], [cudaMalloc],     [], [AC_MSG_FAILURE([CUDA runtime libs not found])])

#check for verbs
AC_ARG_WITH([verbs],
            [AC_HELP_STRING([--with-verbs(=DIR)],
            [Build Infiniband  support, adding DIR/include, DIR/lib, and DIR/lib64 to the search path for headers and libraries])],
            [CFLAGS="-I$with_verbs/include $CFLAGS"
             LDFLAGS="-L$with_verbs/lib64 -L$with_verbs/lib -libverbs $LDFLAGS"],
            [CFLAGS="-I/usr/include $CFLAGS"
             LDFLAGS="-L/usr/lib64 -L/usr/lib -libverbs $LDFLAGS"])

AC_CHECK_HEADER(        [infiniband/verbs.h],  [],[AC_MSG_FAILURE([ibverbs header files not found])])
AC_CHECK_LIB([ibverbs], [ibv_get_device_list], [],[AC_MSG_FAILURE([libibverbs not found]);])
AC_CHECK_MEMBERS([struct ibv_port_attr.active_speed_ex], [], [], [#include <infiniband/verbs.h>])

AC_CHECK_DECLS([IBV_ACCESS_RELAXED_ORDERING, IBV_QPF_GRH_REQUIRED, ibv_reg_dmabuf_mr, ibv_query_ece, ibv_set_ece], [], [],
                    [[#include <infiniband/verbs.h>]])

#
# mlx5 DV support
#
AC_ARG_WITH([mlx5-dv],
            [AC_HELP_STRING([--with-mlx5-dv], [Compile with mlx5 Direct Verbs
                support. Direct Verbs (DV) support provides additional
                acceleration capabilities that are not available in a
                regular mode.])])

AS_IF([test "x$with_mlx5_dv" != xno], [
    AC_CHECK_LIB([mlx5], [mlx5dv_query_device],
                    [AC_SUBST(LIB_MLX5, [-lmlx5])],
                    [with_mlx5_dv=no], [-libverbs])

    AS_IF([test "x$with_mlx5_dv" != xno], [
        AC_CHECK_HEADERS([infiniband/mlx5dv.h],
                         [with_mlx5_dv=yes
                         mlx5_include=mlx5dv.h], [], [ ])])

        AS_IF([test "x$with_mlx5_dv" = "xyes" ], [
        AC_CHECK_DECLS([
                       mlx5dv_query_device,
                       mlx5dv_create_qp,
                       mlx5dv_is_supported,
                       mlx5dv_get_data_direct_sysfs_path,
                       mlx5dv_reg_dmabuf_mr],
                       [], [], [[#include <infiniband/mlx5dv.h>]])])
    ],
    [
        with_mlx5_dv=no
    ])

AS_IF([test "x$with_mlx5_dv" = "xyes" ], [LDFLAGS="$LDFLAGS -lmlx5"], [])


# check for ucx
AM_CONDITIONAL([HAVE_UCX_PLUGIN], [false])
m4_include([m4/ucx.m4])
CHECK_UCX
AC_MSG_RESULT([UCX support: $ucx_happy])

# check for sharp
AM_CONDITIONAL([HAVE_SHARP_PLUGIN], [false])
m4_include([m4/sharp.m4])
CHECK_SHARP
AC_MSG_RESULT([SHARP support: $sharp_happy])

#check for mlx5dv
AM_CONDITIONAL([HAVE_MLX5_DV], [test "x$with_mlx5_dv" = xyes])

#check for required headers
AC_CHECK_HEADERS([limits.h stdlib.h string.h unistd.h], [],
                 [AC_MSG_FAILURE([unable to find required headers])])

AC_CONFIG_FILES([Makefile
                 src/Makefile
                 nccl-rdma-sharp-plugins.spec
                 nccl-rdma-sharp-plugins.pc
                 debian/changelog
                 debian/control
                 debian/nccl-rdma-sharp-plugins.postinst
                 debian/nccl-rdma-sharp-plugins.prem
                 debian/rules
                ])
AC_OUTPUT

echo "NCCL-RDMA-SHARP Plugin has been configured."
