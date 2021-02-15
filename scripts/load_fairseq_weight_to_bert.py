import argparse
from collections import namedtuple
from typing import Dict
from pathlib import Path

import torch


def main(args: argparse.Namespace):
    bert_weight_path: str = args.bert_weight_path
    transformer_weight_path: str = args.transformer_weight_path
    result_path: str = args.result_path

    bert_weight = torch.load(bert_weight_path)
    transformer_weight = torch.load(transformer_weight_path)
    transformer_weight = transformer_weight["model"]
    rename_transformer_weight(transformer_weight)

    # Assert keys
    bert_keys = set(bert_weight.keys())
    transformer_keys = set(transformer_weight.keys())
    enc_transformer_keys = {x for x in transformer_keys if "decoder" not in x}
    diff = enc_transformer_keys - bert_keys
    assert diff == {"bert.encoder.version"}, f"diff = {diff}"

    update_dict = {key: value for key, value in transformer_weight.items() if key in bert_keys}

    # Overwrite weights
    for key, value in update_dict.items():
        bert_weight[key] = value

    # Make dir if not exist
    result_path: Path = Path(result_path)
    result_path.parent.mkdir(exist_ok=True, parents=True)

    torch.save(bert_weight, result_path)


def rename_transformer_weight(transformer_weight: Dict[str, torch.Tensor]) -> None:
    """Rename bert weight's key such that it fits fairseq's weight

    Args:
        bert_weight (Dict[str, torch.Tensor]): bert weight to rename
    """
    Translations = namedtuple("Translations", ["bert_name", "transformer_name"])
    bert2fairseq_list = [
        Translations("embeddings.word_embeddings", "embed_tokens"),
        Translations("embeddings.position_embeddings", "embed_positions"),
        Translations("embeddings.token_type_embeddings", "token_type_embeddings"),
        Translations("embeddings.LayerNorm", "layernorm_embedding"),
        Translations("bert.encoder", "encoder"),
        # Translations("bert", "encoder"),
        # Some of
        Translations("bert.embed_tokens", "encoder.embed_tokens"),
        Translations("bert.embed_positions", "encoder.embed_positions"),
        Translations("bert.token_type_embeddings", "encoder.token_type_embeddings"),
        Translations("bert.layernorm_embedding", "encoder.layernorm_embedding"),
        Translations("attention.self.query", "self_attn.q_proj"),
        Translations("attention.self.key", "self_attn.k_proj"),
        Translations("attention.self.value", "self_attn.v_proj"),
        Translations("attention.output.dense", "self_attn.out_proj"),
        Translations("attention.output.LayerNorm", "self_attn_layer_norm"),
        Translations("intermediate.dense", "fc1"),
        Translations("output.dense", "fc2"),
        Translations("output.LayerNorm", "final_layer_norm"),
        Translations(".layer.", ".layers."),
    ]
    bert2fairseq_list.sort(key=lambda x: -len(x.transformer_name))
    mapping_list = []
    orig_bert_keys = list(transformer_weight.keys())
    for key in orig_bert_keys:
        orig = key
        result = key
        for (target, source) in bert2fairseq_list:
            result = result.replace(source, target)
        mapping_list.append((orig, result))
    for (orig, result) in mapping_list:
        transformer_weight[result] = transformer_weight.pop(orig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load encoder's weight from transformer")
    parser.add_argument('bert_weight_path', metavar='BERT_WEIGHT_PATH', help="Path to bert weight")
    parser.add_argument('transformer_weight_path', metavar='TRANSFORMER_WEIGHT_PATH', help="Path to transformer weight")
    parser.add_argument('result_path', metavar='BERT_WEIGHT_PATH', help="Path to write result weight. e.g. /path/to/hoge/pytorch_model.bin")
    args = parser.parse_args()

    main(args)
