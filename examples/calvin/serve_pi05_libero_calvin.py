"""Serve original pi0.5-LIBERO weights with the paired CALVIN preprocessing."""

import argparse
import logging
from pathlib import Path

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import checkpoints
from openpi.training import config as training_config


ASSET_ID = "Traly/calvin_abc_d-lerobot"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=".openpi_cache/openpi-assets/checkpoints/pi05_libero",
        help="Released pi05_libero checkpoint directory.",
    )
    parser.add_argument(
        "--calvin-norm-checkpoint",
        default="checkpoints/pi05_calvin_lora/calvin_abc_d_lora/29999",
        help="Fine-tuned checkpoint whose CALVIN normalization statistics are reused.",
    )
    parser.add_argument("--port", default=8000, type=int)
    return parser.parse_args()


def main(args):
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    norm_checkpoint = Path(args.calvin_norm_checkpoint).expanduser().resolve()
    if not (checkpoint / "params").is_dir():
        raise FileNotFoundError("Missing pi05_libero params: %s" % (checkpoint / "params"))
    norm_dir = norm_checkpoint / "assets"
    if not norm_dir.is_dir():
        raise FileNotFoundError("Missing fine-tuned checkpoint assets: %s" % norm_dir)

    config = training_config.get_config("pi05_libero_calvin_eval")
    norm_stats = checkpoints.load_norm_stats(norm_dir, ASSET_ID)
    policy = policy_config.create_trained_policy(
        config,
        checkpoint,
        norm_stats=norm_stats,
    )
    logging.info("Original weights: %s", checkpoint)
    logging.info("CALVIN normalization source: %s", norm_checkpoint)
    logging.info("Controlled input: CALVIN cameras, 15-D state carrier, 7-D relative actions")
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(parse_args())
