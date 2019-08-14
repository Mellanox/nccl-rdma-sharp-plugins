#!/bin/sh

rm -rf autom4te.cache
autoreconf -ivf || exit 1
rm -rf autom4te.cache
