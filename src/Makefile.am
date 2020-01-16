#
# Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

lib_LTLIBRARIES     = libnccl-net.la

libnccl_net_la_CFLAGS   = $(CFLAGS) -I$(top_srcdir)/include -DGNU_SOURCE
libnccl_net_la_LIBADD   = -lsharp_coll -lcudart_static
libnccl_net_la_LDFLAGS   = $(LDFLAGS)

libnccl_net_la_SOURCES = \
	ibvwrap.c \
	utils.c	\
	ib_plugin.c \
	sharp_plugin.c
