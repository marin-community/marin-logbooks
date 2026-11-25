"""Run the five companion tasks after verifying the pilot's archived evidence."""

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from .check_pilot import check_pilot
from .run import launch, launch_parser


if __name__ == "__main__":
    parser = launch_parser()
    parser.add_argument("--graded-pilot", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    print("Verified task 013", check_pilot(StoragePath(args.graded_pilot)), flush=True)
    launch(args, ("001", "003", "012", "015", "028"), "panel")
