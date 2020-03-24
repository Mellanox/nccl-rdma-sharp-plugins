#
# Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#
AC_PREREQ([2.63])

AC_COPYRIGHT([Copyright (C) 2019 Mellanox Technologies Ltd. All rights reserved.])

define([nccl_rdma_sharp_plugins_ver_major], 1)
define([nccl_rdma_sharp_plugins_ver_minor], 0)

AC_INIT([nccl-rdma-sharp-plugins], [nccl_rdma_sharp_plugins_ver_major.nccl_rdma_sharp_plugins_ver_minor], [support@mellanox.com], [],[http://github.com/Mellanox/nccl-rdma-sharp-plugins])

AM_INIT_AUTOMAKE([1.10 foreign tar-ustar subdir-objects])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])
AM_MAINTAINER_MODE
AC_CONFIG_MACRO_DIR([config/m4])

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
AC_PROG_CC_STDC
AC_PROG_CXX
AM_PROG_AS
AC_PROG_SED
AC_PROG_INSTALL
AC_PROG_LIBTOOL
AC_HEADER_STDC
LT_LIB_M

#check for cuda

AC_ARG_WITH([cuda],
            AC_HELP_STRING([--with-cuda=PATH], [Path to non-standard CUDA installation]),
            [AS_IF([test -d $withval/lib64], [cuda_libdir="lib64"], [cuda_libdir="lib"])
             CFLAGS="-I$withval/include $CFLAGS"
             LDFLAGS="-L$withval/$cuda_libdir $LDFLAGS"],
             [CFLAGS="-I/usr/local/cuda/include $CFLAGS"
             LDFLAGS="-L/usr/local/cuda/$cuda_libdir $LDFLAGS"])

AC_CHECK_HEADER([cuda_runtime.h], [],
                [AC_MSG_FAILURE([cuda runtime header files not found])])
AC_CHECK_LIB([cudart], [cudaMalloc],
            [],[AC_MSG_FAILURE([cuda runtime libs  not found])])

#check for verbs
AC_ARG_WITH([verbs],
        [AC_HELP_STRING([--with-verbs(=DIR)],
            [Build Infiniband  support, adding DIR/include, DIR/lib, and DIR/lib64 to the search path for headers and libraries])],
        [CFLAGS="-I$with_verbs/include $CFLAGS"
         LDFLAGS="-L$with_verbs/lib64 -L$with_verbs/lib -libverbs $LDFLAGS"],
        [CFLAGS="-I/usr/include $CFLAGS"
         LDFLAGS="-L/usr/lib64 -L/usr/lib -libverbs $LDFLAGS"])

AC_CHECK_HEADER([infiniband/verbs.h], [],
                [AC_MSG_FAILURE([ibverbs header files not found])])
AC_CHECK_LIB([ibverbs], [ibv_get_device_list],
            [],[AC_MSG_FAILURE([libibverbs not found]);])
AC_CHECK_DECLS(IBV_ACCESS_RELAXED_ORDERING, [], [],
                    [[#include <infiniband/verbs.h>]])

ucx_happy="no"
AC_ARG_WITH([ucx], [AS_HELP_STRING([--with-ucx=(DIR)], [Enable the use of UCX (default is guess).])], [], [with_ucx=guess])
AS_IF([test "x$with_ucx" != "xno"],
[
    save_CPPFLAGS="$CPPFLAGS"
    save_CFLAGS="$CFLAGS"
    save_LDFLAGS="$LDFLAGS"
    AS_IF([test ! -z "$with_ucx" -a "x$with_ucx" != "xyes" -a "x$with_ucx" != "xguess"],
    [
        ucx_check_ucx_dir="$with_ucx"
        AS_IF([test -d "$with_ucx/lib64"],
        [
            libsuff="64"
        ],
        [
            libsuff=""
        ])
        ucx_check_ucx_libdir="$with_ucx/lib$libsuff"
        CPPFLAGS="-I$with_ucx/include $save_CPPFLAGS"
        LDFLAGS="-L$ucx_check_ucx_libdir $save_LDFLAGS"
    ])
    AS_IF([test ! -z "$with_ucx_libdir" -a "x$with_ucx_libdir" != "xyes"],
    [
        ucx_check_ucx_libdir="$with_ucx_libdir"
        LDFLAGS="-L$ucx_check_ucx_libdir $save_LDFLAGS"
    ])
    AC_CHECK_HEADERS([ucp/api/ucp.h],
    [
        AC_CHECK_LIB([ucp], [ucp_tag_send_nb],
        [
            ucx_happy="yes"
        ],
        [
            ucx_happy="no"
        ], [-luct -lucm -lucs])
    ],
    [
        ucx_happy="no"
    ])
    CFLAGS="$save_CFLAGS"
    CPPFLAGS="$save_CPPFLAGS"
    LDFLAGS="$save_LDFLAGS"
    AS_IF([test "x$ucx_happy" = "xyes"],
    [
        AS_IF([test ! -z "$ucx_check_ucx_dir"],
        [
            AC_SUBST(UCX_CPPFLAGS, "-I$ucx_check_ucx_dir/include/")
        ])
        AS_IF([test ! -z "$ucx_check_ucx_libdir"],
        [
            AC_SUBST(UCX_LDFLAGS, "-L$ucx_check_ucx_libdir")
        ])
        AC_SUBST(UCX_LIBADD, "-lucp -lucs -lucm -luct")
        AC_DEFINE(HAVE_UCX_PLUGIN, [1], [Define if using UCX plugin])
    ],
    [
        AC_MSG_WARN(["UCX packages cannot be found"])
    ])
],
[
    AC_MSG_WARN(["UCX plugin was explicitly disabled"])
])
AM_CONDITIONAL([HAVE_UCX], [test "x$ucx_happy" != xno])

# check for sharp
AC_ARG_WITH([sharp],
            AC_HELP_STRING([--with-sharp=PATH], [Path to non-standard SHARP installation]),
             [CFLAGS="-I$withval/include $CFLAGS"
              LDFLAGS="-L$withval/lib $LDFLAGS"],
             [AS_IF([test -d "$HPCX_SHARP_DIR"],
                    [CFLAGS="-I$HPCX_SHARP_DIR/include $CFLAGS"
                     LDFLAGS="-L$HPCX_SHARP_DIR/lib $LDFLAGS"],
                     [CFLAGS="-I/opt/mellanox/sharp/include $CFLAGS"
                     LDFLAGS="-L/opt/mellanox/sharp/lib $LDFLAGS"])])

AC_CHECK_HEADER([sharp/api/sharp.h], [],
                [AC_MSG_FAILURE([sharp.h header file not found])])

AC_CHECK_LIB([sharp_coll], [sharp_coll_init],
            [],[AC_MSG_FAILURE([libsharp_coll not found]);])

AC_CHECK_HEADERS([limits.h stdlib.h string.h unistd.h cuda_runtime.h], [],[
    AC_MSG_ERROR([unable to find required headers])
])

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


