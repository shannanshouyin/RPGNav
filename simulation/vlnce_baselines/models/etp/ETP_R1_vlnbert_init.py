import torch
from habitat import logger
from .checkpoint_compat import rename_rp_state_dict


def _filter_mismatched_stgasa_weights(state_dict, config):
    if not state_dict:
        return state_dict
    expected_rows = int(getattr(config, "stgasa_time_clip_steps", 20)) * 2 + 1
    expected_bins = int(getattr(config, "rp_planner_num_bins", 10))
    removed = []
    old_fragments = (
        "st_dist_embed", "st_mlp", "st_rel_mlp", "st_phi",
        "st_psi", "st_lambda_v", "st_time_bucket",
    )
    for key in list(state_dict.keys()):
        if any(fragment in key for fragment in old_fragments):
            removed.append((key, "old_stgasa_v1", None))
            state_dict.pop(key)
            continue
        if key.endswith("st_time_embedding.weight") and state_dict[key].shape[0] != expected_rows:
            removed.append((key, tuple(state_dict[key].shape), (expected_rows, state_dict[key].shape[1])))
            state_dict.pop(key)
            continue
        if key.endswith("rp_progress_head.weight") and state_dict[key].shape[0] != expected_bins:
            removed.append((key, tuple(state_dict[key].shape), (expected_bins, state_dict[key].shape[1])))
            state_dict.pop(key)
            continue
        if key.endswith("rp_progress_head.bias") and state_dict[key].shape[0] != expected_bins:
            removed.append((key, tuple(state_dict[key].shape), (expected_bins,)))
            state_dict.pop(key)
    if removed:
        logger.warning(
            "STGASA checkpoint keys ignored/reinitialized: %s",
            removed[:12],
        )
    return state_dict


def get_tokenizer(args):
    from transformers import AutoTokenizer
    if args.dataset == 'rxr' or args.tokenizer == 'xlm':
        cfg_name = 'bert_config/xlm-roberta-base'
    else:
        cfg_name = 'bert_config/bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer

def get_vlnbert_models(config=None, dropout_rate=0.1):
    
    from transformers import PretrainedConfig
    from vlnce_baselines.models.etp.ETP_R1_vilmodel_cmt import GlocalTextPathNavCMT

    model_class = GlocalTextPathNavCMT

    model_name_or_path = config.pretrained_path
    new_ckpt_weights = {}
    keywords = ['graph_query_text', 'graph_attentioned_txt_embeds_transform', 'global_sap_head', 'rp_planner', 'rp_progress_head', 'rp_progress_residual', 'rp_progress_beta']
    if model_name_or_path:
        ckpt_weights = rename_rp_state_dict(torch.load(model_name_or_path, map_location='cpu'))
        for k, v in ckpt_weights.items():
            clean_k = k[7:] if k.startswith('module.') else k
            if clean_k.startswith('bert.'):
                new_ckpt_weights[clean_k[5:]] = v
            new_ckpt_weights[clean_k] = v
            if k.startswith('module'):
                new_ckpt_weights[k[7:]] = v
            if any(key in k for key in keywords):
                new_ckpt_weights['bert.' + k] = v
    
    cfg_name = 'bert_config/xlm-roberta-base'
    vis_config = PretrainedConfig.from_pretrained(cfg_name)

    vis_config.type_vocab_size = 2

    vis_config.max_action_steps = 100
    vis_config.image_feat_size = 512
    vis_config.use_depth_embedding = config.use_depth_embedding
    vis_config.depth_feat_size = 128
    vis_config.angle_feat_size = 4

    vis_config.num_l_layers = 12
    vis_config.num_pano_layers = 2
    vis_config.num_x_layers = 4
    vis_config.graph_sprels = config.use_sprels
    vis_config.glocal_fuse = 'global'

    vis_config.fix_lang_embedding = config.fix_lang_embedding
    vis_config.fix_pano_embedding = config.fix_pano_embedding

    vis_config.update_lang_bert = not vis_config.fix_lang_embedding
    vis_config.output_attentions = True

    vis_config.pred_head_dropout_prob = dropout_rate
    vis_config.hidden_dropout_prob = dropout_rate
    vis_config.attention_probs_dropout_prob = dropout_rate

    vis_config.use_lang2visn_attn = True

    vis_config.max_txt_task_embeddings = 4
    vis_config.max_gmap_task_embeddings = 3
    vis_config.stgasa = getattr(config, "stgasa", False)
    st_cfg = getattr(config, "STGASA", None)
    vis_config.stgasa_time_clip_steps = getattr(st_cfg, "time_clip_steps", 20) if st_cfg is not None else 20
    vis_config.stgasa_relation_hidden_size = getattr(st_cfg, "relation_hidden_size", 128) if st_cfg is not None else 128
    vis_config.stgasa_gate_init = getattr(st_cfg, "gate_init", 0.0) if st_cfg is not None else 0.0
    vis_config.rp_planner = getattr(config, "rp_planner", False)
    rpp_cfg = getattr(config, "RP_PLANNER", None)
    vis_config.rp_planner_num_bins = getattr(rpp_cfg, "num_bins", 10) if rpp_cfg is not None else 10
    vis_config.rp_planner_max_label_dist = getattr(rpp_cfg, "max_label_dist", 3.0) if rpp_cfg is not None else 3.0
    vis_config.rp_planner_beta_init = getattr(rpp_cfg, "beta_init", 0.0) if rpp_cfg is not None else 0.0
    vis_config.rp_planner_use_progress_score = getattr(rpp_cfg, "use_progress_score", True) if rpp_cfg is not None else True
    vis_config.rp_planner_use_type_embedding = getattr(rpp_cfg, "use_type_embedding", True) if rpp_cfg is not None else True
    vis_config.rp_planner_use_step_embedding = getattr(rpp_cfg, "use_step_embedding", True) if rpp_cfg is not None else True
    vis_config.rp_planner_type_embedding_mode = getattr(rpp_cfg, "type_embedding_mode", "learned") if rpp_cfg is not None else "learned"
    vis_config.rp_planner_step_embedding_mode = getattr(rpp_cfg, "step_embedding_mode", "learned") if rpp_cfg is not None else "learned"
    new_ckpt_weights = _filter_mismatched_stgasa_weights(new_ckpt_weights, vis_config)

    visual_model = model_class.from_pretrained(
        pretrained_model_name_or_path=None, 
        config=vis_config, 
        state_dict=new_ckpt_weights)
    return visual_model
