#!/bin/bash
# cw.sh METHOD [args...]  -> gdbus call into the test shell
export DBUS_SESSION_BUS_ADDRESS=$(cat /tmp/cwtest/bus)
m=$1; shift
gdbus call --session -d io.github.bogdanstamenovic.Classwatch -o /io/github/bogdanstamenovic/Classwatch -m io.github.bogdanstamenovic.Classwatch.$m "$@"
