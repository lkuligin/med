-- lsyncd configuration: mirror this repository to the GPU box.
--
--   lsyncd gdanschin_runtime/sync/lsyncd-med.lua
--   (or use ./gdanschin_runtime/sync/sync.sh, which preflights first)
--
-- One-way only: local -> remote. Files created on the remote by running the
-- project are protected by two independent mechanisms:
--   1. delete = false      - rsync never removes remote-only files
--   2. excludeFrom         - output paths are never transferred at all
-- Both are needed: (1) alone would still let a stale local results_*.json
-- overwrite a fresh remote one.

settings {
    logfile    = "/tmp/lsyncd-med.log",
    statusFile = "/tmp/lsyncd-med.status",
    nodaemon   = true,   -- run in the foreground; Ctrl-C stops the sync
    insist     = true,   -- keep retrying instead of exiting if the box is down
}

local home     = os.getenv( "HOME" )
local here     = os.getenv( "MED_SYNC_CONF_DIR" ) or ( home .. "/Projects/med/gdanschin_runtime/sync" )
local source   = os.getenv( "MED_SYNC_SOURCE" )   or ( home .. "/Projects/med" )
local identity = os.getenv( "MED_SYNC_IDENTITY" ) or ( home .. "/.ssh/g.danschin" )

sync {
    default.rsync,

    source = source,
    target = "gdanschin@gpu.example.com:/home/gdanschin/Projects/med/",

    delay  = 1,       -- seconds to batch edits before pushing
    delete = false,   -- never delete on the remote; see header

    excludeFrom = here .. "/rsync-exclude.txt",

    rsync = {
        archive  = true,
        compress = true,
        rsh      = "ssh -i " .. identity,
    },
}

-- Optional hardening: adding rsync.update = true makes rsync skip any file that
-- is NEWER on the remote. Left off on purpose - it would also silently stop
-- propagating local edits whenever the remote clock runs ahead, which is a
-- much worse failure than the one it prevents.
