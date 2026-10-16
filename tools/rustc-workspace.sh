#!/bin/sh
# Cargo hashes this wrapper's resolved path into workspace artifact identities.
# Each checkout gets distinct application artifacts while registry dependencies
# remain shared. Compilation itself is unchanged.
exec "$@"
