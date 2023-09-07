#
# Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See file LICENSE for terms.
#
AC_PREREQ([2.63])

AC_COPYRIGHT([Copyright (c) 2019, NVIDIA CORPORATION & AFFILIATES. All rights reserved.])

define([nccl_rdma_sharp_plugins_ver_major], 2)
define([nccl_rdma_sharp_plugins_ver_minor], 5)

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
	CFLAGS="$CFLAGS -O0 -g3 -Werror"
else
	CFLAGS="$CFLAGS -O3 -DNDEBUG -Werror"
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

AC_CHECK_DECLS([IBV_ACCESS_RELAXED_ORDERING, IBV_QPF_GRH_REQUIRED, ibv_reg_dmabuf_mr], [], [],
                    [[#include <infiniband/verbs.h>]])

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
