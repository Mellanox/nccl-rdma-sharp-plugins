#
# Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

lib_LTLIBRARIES     = libnccl-net.la

libnccl_net_la_CFLAGS   = $(CFLAGS) -I$(top_srcdir)/include -DGNU_SOURCE
libnccl_net_la_LIBADD   = -lsharp_coll -lcudart_static
libnccl_net_la_LDFLAGS  = $(LDFLAGS)

libnccl_net_la_SOURCES = \
	ibvwrap.c \
	utils.c	\
	p2p_plugin.c \
	ib_plugin.c \
	sharp_plugin.c

if HAVE_UCX
libnccl_net_la_CPPFLAGS  = -DHAVE_UCX_PLUGIN $(UCX_CPPFLAGS)
libnccl_net_la_LIBADD   += $(UCX_LIBADD)
libnccl_net_la_LDFLAGS  += $(UCX_LDFLAGS) 
libnccl_net_la_SOURCES  += ucx_plugin.c
endif
