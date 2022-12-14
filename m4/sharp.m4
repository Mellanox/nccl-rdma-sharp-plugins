#
# Copyright (c) 2001-2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

AC_DEFUN([CHECK_SHARP],[

AS_IF([test "x$sharp_checked" != "xyes"],[

    sharp_happy="no"

    AC_ARG_WITH([sharp],
            [AS_HELP_STRING([--with-sharp=(DIR)], [Enable the use of SHARP (default is guess).])],
            [], [with_sharp=guess])

    AS_IF([test "x$with_sharp" != "xno"],
    [
        save_CPPFLAGS="$CPPFLAGS"
        save_CFLAGS="$CFLAGS"
        save_LDFLAGS="$LDFLAGS"

        AS_IF([test ! -z "$with_sharp" -a "x$with_sharp" != "xyes" -a "x$with_sharp" != "xguess"],
        [
            check_sharp_dir="$with_sharp"
            check_sharp_libdir="$with_sharp/lib"
            CPPFLAGS="-I$with_sharp/include $save_CPPFLAGS"
            LDFLAGS="-L$check_sharp_libdir $save_LDFLAGS"
        ])

        AS_IF([test "x$check_sharp_dir" = "x" -a "x$HPCX_SHARP_DIR" != "x"],
        [
            check_sharp_dir="$HPCX_SHARP_DIR"
            check_sharp_libdir="$HPCX_SHARP_DIR/lib"
            CPPFLAGS="-I$check_sharp_dir/include $save_CPPFLAGS"
            LDFLAGS="-L$check_sharp_libdir $save_LDFLAGS"
        ])

        AS_IF([test "x$check_sharp_dir" = "x" -a -d "/opt/mellanox/sharp/"],
        [
            check_sharp_dir="/opt/mellanox/sharp/"
            check_sharp_libdir="/opt/mellanox/sharp/lib"
            CPPFLAGS="-I$check_sharp_dir/include $save_CPPFLAGS"
            LDFLAGS="-L$check_sharp_libdir $save_LDFLAGS"
        ])


        AS_IF([test ! -z "$with_sharp_libdir" -a "x$with_sharp_libdir" != "xyes"],
        [
            check_sharp_libdir="$with_sharp_libdir"
            LDFLAGS="-L$check_sharp_libdir $save_LDFLAGS"
        ])

        AC_CHECK_HEADERS([sharp/api/sharp_coll.h],
        [
            AC_CHECK_LIB([sharp_coll], [sharp_coll_init],
            [
                sharp_happy="yes"
            ],
            [
                sharp_happy="no"
            ])
        ],
        [
            sharp_happy="no"
        ])

        AS_IF([test "x$sharp_happy" = "xyes"],
        [
            AS_IF([test "x$check_sharp_dir" != "x"],
            [
                AC_MSG_RESULT([SHARP dir: $check_sharp_dir])
                AC_SUBST(SHARP_CPPFLAGS, "-I$check_sharp_dir/include/")
            ])

            AS_IF([test "x$check_sharp_libdir" != "x"],
            [
                AC_SUBST(SHARP_LDFLAGS, "-L$check_sharp_libdir")
            ])

            AC_SUBST(SHARP_LIBADD, "-lsharp_coll")
            AC_CHECK_DECLS([SHARP_DTYPE_BFLOAT16], [AC_DEFINE([HAVE_SHARP_DTYPE_BFLOAT16_UINT8_INT8], 1,
                                                    [SHARP v3 datatypes : bfloat16, uint8, int8])], [],
                           [[#include <sharp/api/sharp_coll.h>]])
            AC_CHECK_DECLS([sharp_coll_reg_mr_v2], [], [], [[#include <sharp/api/sharp_coll.h>]])

        ],
        [
            AS_IF([test "x$with_sharp" != "xguess"],
            [
                AC_MSG_ERROR([SHARP support is requested but SHARP packages cannot be found])
            ],
            [
                AC_MSG_WARN([SHARP not found])
            ])
        ])

        CFLAGS="$save_CFLAGS"
        CPPFLAGS="$save_CPPFLAGS"
        LDFLAGS="$save_LDFLAGS"

    ],
    [
        AC_MSG_WARN([SHARP was explicitly disabled])
    ])

    sharp_checked=yes
    AM_CONDITIONAL([HAVE_SHARP_PLUGIN], [test "x$sharp_happy" != xno])
])

])
