# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
import argparse
import os
import os.path as osp
import sys
import tempfile

import torch

current_path = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.dirname(current_path))

# TODO: Move to config file
import sscma.datasets  # noqa
import sscma.engine  # noqa
import sscma.evaluation  # noqa
import sscma.models  # noqa
import sscma.visualization  # noqa


def _build_teacher_for_kd(cfg, runner_model):
    from mmengine.config import Config
    from sscma.utils import load_config
    from mmengine.runner.checkpoint import load_checkpoint
    from sscma.registry import MODELS
    teacher_cfg_info = cfg.get('teacher', None)
    if teacher_cfg_info is None:
        return None
    try:
        tmp_dir = tempfile.mkdtemp()
        teacher_cfg_data = load_config(teacher_cfg_info['cfg'], folder=tmp_dir, cfg_options={"data_root": cfg.get('data_root', None)})
        teacher_raw = Config.fromfile(teacher_cfg_data)
    except Exception as e:
        print(f'[KD] Failed to load teacher config: {e}')
        return None

    teacher_model = MODELS.build(teacher_raw.model)
    ckpt_path = teacher_cfg_info.get('checkpoint', None)
    import pdb; pdb.set_trace()
    if ckpt_path and os.path.exists(ckpt_path):
        load_checkpoint(teacher_model, ckpt_path, map_location='cpu')
        print(f'[KD] Loaded teacher checkpoint: {ckpt_path}')
    else:
        print(f'[KD] Warning: teacher checkpoint not found: {ckpt_path}')

    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    # Align to same device as student later in runner hook
    return teacher_model


class _DistillManager:
    def __init__(self, cfg_kd, student, teacher):
        self.cfg = cfg_kd
        self.student = student
        self.teacher = teacher
        self.T = cfg_kd.get('temperature', 4.0)
        self.obj_thr = cfg_kd.get('obj_thr', 0.3)
        self.max_pos = cfg_kd.get('max_pos_per_level', 0)
        self.start_epoch = cfg_kd.get('start_epoch', 0)
        self.loss_builders = {}
        from sscma.registry import MODELS as _RM
        losses_cfg = cfg_kd.get('losses', {})
        for k, v in losses_cfg.items():
            if isinstance(v, dict):
                self.loss_builders[k] = _RM.build(v)
        self.built = True

    def _gather_raw_outputs(self, head_module, feats):
        # head_module is YOLOV5Head
        return head_module.head_module(feats)  # tuple(cls_scores, bbox_preds, objectnesses)

    def _reshape_for_mask(self, cls_scores, bbox_preds, obj_scores, num_classes, num_base_priors):
        # reshape to structured tensors
        reshaped = []
        for cs, bp, ob in zip(cls_scores, bbox_preds, obj_scores):
            B, _, H, W = cs.shape
            cs_r = cs.view(B, num_base_priors, num_classes, H, W)
            bp_r = bp.view(B, num_base_priors, 4, H, W)
            ob_r = ob.view(B, num_base_priors, 1, H, W)
            reshaped.append((cs_r, bp_r, ob_r))
        return reshaped

    @torch.no_grad()
    def _build_mask(self, t_levels):
        masks = []
        for (t_cls, t_bbox, t_obj) in t_levels:
            prob = torch.sigmoid(t_obj)  # B,A,1,H,W
            mask = prob.squeeze(2) > self.obj_thr  # B,A,H,W
            if self.max_pos > 0:
                B, A, H, W = mask.shape
                flat = prob.view(B, A * H * W)
                vals, idxs = flat.topk(min(self.max_pos, flat.shape[1]), dim=1)
                topk_mask = torch.zeros_like(flat, dtype=torch.bool)
                topk_mask.scatter_(1, idxs, True)
                topk_mask = topk_mask.view(B, A, H, W)
                mask = mask | topk_mask
            masks.append(mask)
        return masks

    def _select(self, s, t, mask):
        # s,t shapes (B,A,C,H,W) or (B,A,1,H,W) or (B,A,4,H,W)
        if mask.sum() == 0:
            return s.view(0, s.shape[2]), t.view(0, t.shape[2])
        if s.dim() == 5:
            s_sel = s.permute(0,1,3,4,2)[mask]  # (N,C)
            t_sel = t.permute(0,1,3,4,2)[mask]
        else:
            raise RuntimeError('Unexpected tensor rank in distill select')
        return s_sel, t_sel

    def compute(self, epoch, student_feats_tuple, teacher_feats_tuple, student_head, teacher_head):
        device = student_feats_tuple[0].device
        zeros = lambda: torch.tensor(0.0, device=device)
        if epoch < self.start_epoch:
            return dict(kd_loss_cls=zeros(), kd_loss_obj=zeros(), kd_loss_bbox=zeros(), kd_loss_feat=zeros())

        with torch.cuda.amp.autocast(enabled=torch.is_autocast_enabled()):
            with torch.no_grad():
                t_raw = teacher_head.head_module(student_feats_tuple) if teacher_feats_tuple is None else teacher_head.head_module(teacher_feats_tuple)
            s_raw = student_head.head_module(student_feats_tuple)

        # Unpack raw outputs
        s_cls, s_bbox, s_obj = s_raw
        t_cls, t_bbox, t_obj = t_raw
        num_classes = student_head.num_classes
        num_base_priors = student_head.num_base_priors

        s_levels = self._reshape_for_mask(s_cls, s_bbox, s_obj, num_classes, num_base_priors)
        t_levels = self._reshape_for_mask(t_cls, t_bbox, t_obj, num_classes, num_base_priors)
        masks = self._build_mask(t_levels)

        # Collect selected tensors
        s_cls_all = []
        t_cls_all = []
        s_obj_all = []
        t_obj_all = []
        s_bbox_all = []
        t_bbox_all = []
        for (s_c,s_b,s_o),(t_c,t_b,t_o),m in zip(s_levels, t_levels, masks):
            sc_sel, tc_sel = self._select(s_c, t_c, m)
            so_sel, to_sel = self._select(s_o, t_o, m)
            sb_sel, tb_sel = self._select(s_b, t_b, m)
            s_cls_all.append(sc_sel)
            t_cls_all.append(tc_sel)
            s_obj_all.append(so_sel)
            t_obj_all.append(to_sel)
            s_bbox_all.append(sb_sel)
            t_bbox_all.append(tb_sel)
        if len(s_cls_all) == 0:
            return dict(kd_loss_cls=zeros(), kd_loss_obj=zeros(), kd_loss_bbox=zeros(), kd_loss_feat=zeros())

        s_cls_cat = torch.cat(s_cls_all, 0)
        t_cls_cat = torch.cat(t_cls_all, 0)
        s_obj_cat = torch.cat(s_obj_all, 0)
        t_obj_cat = torch.cat(t_obj_all, 0)
        s_bbox_cat = torch.cat(s_bbox_all, 0)
        t_bbox_cat = torch.cat(t_bbox_all, 0)

        losses = {}
        # CLS KD
        if 'cls' in self.loss_builders:
            losses['kd_loss_cls'] = self.loss_builders['cls'](s_cls_cat, t_cls_cat)
        else:
            losses['kd_loss_cls'] = zeros()
        # OBJ KD
        if 'obj' in self.loss_builders:
            losses['kd_loss_obj'] = self.loss_builders['obj'](s_obj_cat, t_obj_cat)
        else:
            losses['kd_loss_obj'] = zeros()
        # BBOX KD
        if 'bbox' in self.loss_builders:
            losses['kd_loss_bbox'] = self.loss_builders['bbox'](s_bbox_cat, t_bbox_cat)
        else:
            losses['kd_loss_bbox'] = zeros()
        # Feature KD
        if 'feat' in self.loss_builders:
            losses['kd_loss_feat'] = self.loss_builders['feat'](student_feats_tuple, teacher_feats_tuple)
        else:
            losses['kd_loss_feat'] = zeros()
        return losses


def parse_args():
    from mmengine.config import DictAction

    parser = argparse.ArgumentParser(description='Train sscma models')

    # common configs
    parser.add_argument('config', type=str, help='the model config file path')
    parser.add_argument(
        '--work_dir',
        '--work-dir',
        type=str,
        default=None,
        help='the directory to save logs and models',
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision during training (https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)',
    )
    parser.add_argument(
        '--auto_scale_lr',
        '--auto-scale-lr',
        action='store_true',
        default=False,
        help='enable automatic-scale-LR during training',
    )
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='resume training from the checkpoint of the last epoch (or a specified checkpoint path)',
    )
    parser.add_argument(
        '--no_validate',
        '--no-validate',
        action='store_true',
        default=False,
        help='disable checkpoint evaluation during training',
    )
    parser.add_argument(
        '--launcher',
        type=str,
        default='none',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        help='the job launcher for MMEngine',
    )
    parser.add_argument(
        '--cfg_options',
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help="override some settings in the used config, the key-value pair in 'xxx=yyy' format will be merged into config file",
    )
    parser.add_argument(
        '--local_rank',
        '--local-rank',
        type=int,
        default=0,
        help='set local-rank for PyTorch',
    )
    parser.add_argument(
        '--dynamo_cache_size',
        '--dynamo-cache-size',
        type=int,
        default=None,
        help='set dynamo-cache-size limit for PyTorch',
    )

    # extension
    parser.add_argument(
        '--input_shape',
        '--input-shape',
        type=int,
        nargs='+',
        default=None,
        help='Extension: input data shape for model parameters estimation, e.g. 1 3 224 224',
    )

    return parser.parse_args()


def verify_args(args):
    assert os.path.splitext(args.config)[-1] == '.py', "The config file name should be ended with a '.py' extension"
    assert os.path.exists(args.config), 'The config file does not exist'
    assert args.local_rank >= 0, 'The local-rank should be larger than or equal to 0'
    if args.dynamo_cache_size is not None:
        assert args.dynamo_cache_size > 0, 'The local-rank should be larger than or equal to 0'

    return args


def build_config(args):
    from mmengine.config import Config

    from sscma.utils import load_config

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.dynamo_cache_size is not None:
        torch._dynamo.config.cache_size_limit = args.dynamo_cache_size

    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_data = load_config(args.config, folder=tmp_dir, cfg_options=args.cfg_options)
        cfg = Config.fromfile(cfg_data)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.launcher = args.launcher

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        args.work_dir = cfg.work_dir = os.path.join('work_dirs', os.path.splitext(os.path.basename(args.config))[0])

    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.get('type', 'OptimWrapper')
        assert optim_wrapper in [
            'OptimWrapper',
            'AmpOptimWrapper',
        ], f'automatic-mixed-precision is not supported by {optim_wrapper}'
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    if args.input_shape is None:
        try:
            if 'imgsz' in cfg:
                args.input_shape = [1, 1 if cfg.get('gray', False) else 3, *cfg.imgsz]
            elif 'width' in cfg and 'height' in cfg:
                args.input_shape = [
                    1,
                    1 if cfg.get('gray', False) else 3,
                    cfg.width,
                    cfg.height,
                ]
            elif 'shape' in cfg:
                args.input_shape = cfg.shape

        except Exception as exc:
            raise ValueError('Please specify the input shape') from exc
        print(
            "Using automatically generated input shape (from config '{}'): {}".format(
                os.path.basename(args.config), args.input_shape
            )
        )

    return args, cfg


def main():
    from sscma.utils.analysis import get_model_complexity_info

    args = parse_args()
    args = verify_args(args)
    args, cfg = build_config(args)

    # Build runner first
    if 'runner_type' not in cfg:
        from mmengine.runner import Runner
        runner = Runner.from_cfg(cfg)
        runner.val_evaluator.dataset_meta = runner.val_dataloader.dataset.METAINFO
    else:
        from mmengine.registry import RUNNERS
        runner = RUNNERS.build(cfg)
        runner.val_evaluator.dataset_meta = runner.val_dataloader.dataset.METAINFO

    # Inject KD teacher & manager into runner
    teacher_model = _build_teacher_for_kd(cfg, runner.model)
    if teacher_model is not None:
        teacher_model = teacher_model.to(next(runner.model.parameters()).device)
        kd_manager = _DistillManager(cfg.get('kd', {}), runner.model, teacher_model)
        runner.model._kd_teacher = teacher_model  # attach
        runner.model._kd_manager = kd_manager
        print('[KD] Teacher and KD manager initialized.')
    else:
        runner.model._kd_teacher = None
        runner.model._kd_manager = None

    device = next(runner.model.parameters()).device
    runner.model.eval()
    print(args.input_shape)

    analysis_results = get_model_complexity_info(
        model=runner.model,
        input_shape=tuple(args.input_shape[1:]),
        show_arch=False,
        device=device,
    )
    print(analysis_results['out_table'])
    print('=' * 40)
    print(f"{'Input Shape':^20}:{str(args.input_shape):^20}")
    print(f"{'Model Flops':^20}:{analysis_results['flops_str']:^20}")
    print(f"{'Model Parameters':^20}:{analysis_results['params_str']:^20}")
    print('=' * 40)

    # Patch train_step to include KD
    orig_train_step = runner.model.train_step

    def kd_train_step(data, optim_wrapper):
        if not hasattr(runner.model, '_kd_manager') or runner.model._kd_manager is None:
            return orig_train_step(data, optim_wrapper)
        # Apply data preprocessor like original train_step
        with torch.no_grad():
            processed = runner.model.data_preprocessor(data, training=True)
        batch_inputs = processed['inputs']
        data_samples = processed['data_samples']
        # Student features
        with torch.cuda.amp.autocast(enabled=torch.is_autocast_enabled()):
            student_feats = runner.model.extract_feat(batch_inputs)
        # Teacher features
        with torch.no_grad():
            teacher_feats = runner.model._kd_teacher.extract_feat(batch_inputs)
        # Detection losses
        det_losses = runner.model.bbox_head.loss(student_feats, data_samples)
        # KD losses
        kd_losses = runner.model._kd_manager.compute(
            epoch=getattr(runner, 'epoch', 0),
            student_feats_tuple=student_feats,
            teacher_feats_tuple=teacher_feats,
            student_head=runner.model.bbox_head,
            teacher_head=runner.model._kd_teacher.bbox_head,
        )
        total_loss = sum(det_losses.values()) + sum(kd_losses.values())
        log_vars = {**det_losses, **kd_losses, 'loss': total_loss}
        optim_wrapper.update_params(total_loss)
        return {'loss': total_loss, 'log_vars': log_vars, 'num_samples': len(data_samples)}

    runner.model.train_step = kd_train_step

    runner.train()


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
