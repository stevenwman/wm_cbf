# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer)
sys.path.append(str(pathlib.Path(__file__).parent))

import models
import tools
import torch
import collections  

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro
from tqdm import trange

to_np = lambda x: x.detach().cpu().numpy()

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "WM_CBF_Dubins"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "dreamer"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_trajectory: bool = False
    """whether to save trajectory data into the `videos` folder"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    log_freq: int = 1_000_000
    """logging frequency in terms of environment steps"""

    parallel: bool = True
    eval_every: int = 10_000
    eval_episode_num: int = 10
    log_every: int = 10_000
    reset_every: int =  0
    device: str = 'cuda:0'
    compile: bool = True
    precision: int =  32
    debug: bool =  False
    video_pred_log: bool =  True
    precision: int = 32
    action_repeat: int = 1
    steps: int = 10_000_000

    eval_every: int = 10_000
    log_every: int = 10_000
    #time_limit: int = 1e3
    offline_traindir: str = ''
    offline_evaldir: str = ''
    reset_every: int = 0

    dyn_hidden: int = 512
    dyn_deter: int = 512
    dyn_stoch: int = 32
    dyn_discrete: int = 32
    dyn_rec_depth: int = 1
    dyn_mean_act: str = 'none'
    dyn_std_act: str = 'sigmoid2'
    dyn_min_std: float = 0.1
    units: int = 512
    act: str ='SiLU'
    norm: bool = True
    dyn_scale: float = 0.5
    rep_scale: float = 0.1
    kl_free: float = 1.0
    weight_decay: float = 0.0
    unimix_ratio: float = 0.01
    initial: str = 'learned'


    batch_size: int = 32
    batch_length: int = 16
    train_ratio: int = 64
    model_lr: float = 1e-4
    opt_eps: float = 1e-8
    grad_clip: int = 1000
    dataset_size: int = 1_000_000
    opt: str = 'adam'

    gamma_lx: float = 0.75

    turnRate: float = 1.25
    x_min: float = -1.5
    x_max: float = 1.5
    y_min: float = -1.5
    y_max: float = 1.5
    size: int = 128
    logdir: str = 'logs/dreamer_dubins' 


    encoder: Dict[str, Any] = field(default_factory=lambda:{'mlp_keys': 'obs_state', 'cnn_keys': 'image', 'act': 'SiLU', 'norm': True, 'cnn_depth': 32, 'kernel_size': 4, 'minres': 4, 'mlp_layers': 5, 'mlp_units': 1024, 'symlog_inputs': True})
    decoder: Dict[str, Any] = field(default_factory=lambda:{'mlp_keys': 'obs_state', 'cnn_keys': 'image', 'act': 'SiLU', 'norm': True, 'cnn_depth': 32, 'kernel_size': 4, 'minres': 4, 'mlp_layers': 5, 'mlp_units': 1024, 'cnn_sigmoid': False, 'image_dist': 'mse', 'vector_dist': 'symlog_mse', 'outscale': 1.0})
    actor: Dict[str, Any] = field(default_factory=lambda:{'layers': 2, 'dist': 'normal', 'entropy': 3e-4, 'unimix_ratio': 0.01, 'std': 'learned', 'min_std': 0.1, 'max_std': 1.0, 'temp': 0.1, 'lr': 3e-5, 'eps': 1e-5, 'grad_clip': 100.0, 'outscale': 1.0})
    critic: Dict[str, Any] = field(default_factory=lambda:{'layers': 2, 'dist': 'symlog_disc', 'slow_target': True, 'slow_target_update': 1, 'slow_target_fraction': 0.02, 'lr': 3e-5, 'eps': 1e-5, 'grad_clip': 100.0, 'outscale': 0.0})
    reward_head:  Dict[str, Any] = field(default_factory=lambda:{'layers': 2, 'dist': 'symlog_disc', 'loss_scale': 1.0, 'outscale': 0.0})
    cont_head:  Dict[str, Any] = field(default_factory=lambda:{'layers': 2, 'loss_scale': 1.0, 'outscale': 1.0})
    margin_head:  Dict[str, Any] = field(default_factory=lambda:{'layers': 2, 'loss_scale': 1.0})
    grad_heads: List[str] = field(default_factory=lambda: ['decoder'])

    train_steps: int = 50_000
    dataset_path: str = '/home/kensuke/WM_CBF/wm_cbf/wm_demos128_gap.pkl'
    num_trajs: int = 4000 # 2000 in paper
    num_train_trajs: int = 3800

    discount: int = 1.


def make_dataset(episodes, args):
    generator = tools.sample_episodes(episodes, args.batch_length)
    dataset = tools.from_generator(generator, args.batch_size)
    return dataset
    

class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, args, logger, dataset, expert_dataset=None):
        super(Dreamer, self).__init__()
        self._args = args
        self._should_log = tools.Every(args.log_every)
        batch_steps = args.batch_size * args.batch_length
        self._should_train = tools.Every(batch_steps / args.train_ratio)
        self._should_reset = tools.Every(args.reset_every)
        self._metrics = {}
        # this is update step
        self._step = logger.step // args.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._expert_dataset = expert_dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, args)
        if (
            args.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)


    def pretrain_model_only(self, data):
        metrics = {}
        wm = self._wm
        data = wm.preprocess(data)
        
        # world model reconstruction + KL loss
        with tools.RequiresGrad(wm):
            with torch.amp.autocast("cuda", enabled=wm._use_amp):
                embed = wm.encoder(data)
                # post: z_t, prior: \hat{z}_t
                post, prior = wm.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                # note: kl_loss is already sum of dyn_loss and rep_loss
                kl_loss, kl_value, dyn_loss, rep_loss = wm.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape

                losses = {}
                feat = wm.dynamics.get_feat(post)

                preds = {}
                for name, head in wm.heads.items():
                    if name != "margin":
                        grad_head = name in self._config.grad_heads
                        feat = wm.dynamics.get_feat(post)
                        feat = feat if grad_head else feat.detach()
                        pred = head(feat)
                        if type(pred) is dict:
                            preds.update(pred)
                        else:
                            preds[name] = pred
                # preds is dictionary of all all MLP+CNN keys
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                    
                recon_loss = sum(losses.values())
                model_loss = kl_loss + recon_loss
                metrics = self.pretrain_opt(
                    torch.mean(model_loss), self.pretrain_params
                )

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_loss"] = to_np(kl_loss)
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl_value"] = to_np(torch.mean(kl_value))

        with torch.amp.autocast("cuda", enabled=wm._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(wm.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(wm.dynamics.get_dist(post).entropy())
            )
        

        feat_detached = self.dynamics.get_feat(post).detach()
        safe_data = torch.where(data["failure"] == 0.)
        unsafe_data = torch.where(data["failure"] == 1.)
        safe_dataset = feat_detached[safe_data]
        unsafe_dataset = feat_detached[unsafe_data]

        # gradient penalty head
        with tools.RequiresGrad(self.heads["margin_gp"]):
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                pos = self.heads["margin_gp"](safe_dataset)
                neg = self.heads["margin_gp"](unsafe_dataset)
                #print('gp', pos.shape, neg.shape)
                N = max(pos.numel(), neg.numel())
                gp_loss=torch.tensor(0., device=pos.device)
                if pos.numel() > 0 and neg.numel() > 0:
                    if N > safe_dataset.shape[0]:
                        repeat_times = (N + safe_dataset.shape[0] - 1) // safe_dataset.shape[0]  # Ceiling division
                        safe_repeated = safe_dataset.repeat((repeat_times,) + (1,) * (safe_dataset.dim() - 1))  # Repeat along batch dim
                        indices = torch.randperm(safe_repeated.shape[0], device=safe_dataset.device)[:N]
                        pos_data =  safe_repeated[indices]
                    else:
                        pos_data = safe_dataset
                    if N > unsafe_dataset.shape[0]:
                        repeat_times = (N + unsafe_dataset.shape[0] - 1) // unsafe_dataset.shape[0]  # Ceiling division
                        unsafe_repeated = unsafe_dataset.repeat((repeat_times,) + (1,) * (unsafe_dataset.dim() - 1))  # Repeat along batch dim
                        indices = torch.randperm(unsafe_repeated.shape[0], device=unsafe_dataset.device)[:N]
                        neg_data =  unsafe_repeated[indices]
                    else:
                        neg_data = unsafe_dataset
                    # gradient penalty
                    alpha = torch.rand(pos_data.shape[0], 1, device=pos_data.device)
                    interpolates = alpha * pos_data + (1 - alpha) * neg_data
                    interpolates.requires_grad_(True)
                    disc_interpolates = self.heads["margin_gp"](interpolates)

                    gradients = torch.autograd.grad(
                        outputs=disc_interpolates,
                        inputs=interpolates,
                        grad_outputs=torch.ones_like(disc_interpolates),
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )[0]
                    gradients = gradients.view(pos_data.shape[0], -1)
                    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)

                    gp_loss = ((gradients_norm - 0.1) ** 2).mean()

                

                
                relu_loss = torch.tensor(0., device=pos.device)
                zero_sum_loss = torch.tensor(0., device=pos.device)
                if pos.numel() > 0:
                    pos_mean = pos.mean()
                    zero_sum_loss -= pos_mean
                    relu_loss += torch.relu(gamma - pos).mean()
                if neg.numel() >0:
                    neg_mean = neg.mean()
                    zero_sum_loss += neg_mean
                    relu_loss += torch.relu(gamma + neg).mean()
                print(neg.numel(), pos.numel())
                relu_weight = 100
                gp_weight = 10
                loss = zero_sum_loss + relu_weight * relu_loss + gp_weight * gp_loss
                
                metrics.update(self.margin_gp_opt(loss, self.heads["margin_gp"].parameters()))
                metrics["margin_gp"] = gp_loss.item()
                metrics["sign_loss"] = to_np(relu_loss)
                metrics["gp_loss"] = to_np(gp_loss)

        # no gradient penalty head
        with tools.RequiresGrad(self.heads["margin_nogp"]):
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                pos = self.heads["margin_nogp"](safe_dataset)
                neg = self.heads["margin_nogp"](unsafe_dataset)
                gamma = self._config.gamma_lx
                lx_loss = 0.0
                if pos.numel() > 0:
                    lx_loss += torch.relu(gamma - pos).mean()
                if neg.numel() > 0:
                    lx_loss += torch.relu(gamma + neg).mean()

                metrics["margin_nogp"] = lx_loss.item()
                metrics.update(self.margin_nogp_opt(lx_loss, self.heads["margin_nogp"].parameters()))
    
        metrics = {
            f"model_only_pretrain/{k}": v for k, v in metrics.items()
        }  # Add prefix model_pretrain to all metrics
        self._update_running_metrics(metrics)
        self._maybe_log_metrics()
        self._step += 1
    

if __name__ == "__main__":
    args = tyro.cli(Args)

    # defining environment 
    action_space = gym.spaces.Box(
        low=-args.turnRate, high=args.turnRate, shape=(1,), dtype=np.float32
    )
    bounds = np.array([[args.x_min, args.x_max], [args.y_min, args.y_max], [0, 2 * np.pi]])
    low = bounds[:, 0]
    high = bounds[:, 1]
    midpoint = (low + high) / 2.0
    interval = high - low
    gt_observation_space = gym.spaces.Box(
        np.float32(midpoint - interval/2),
        np.float32(midpoint + interval/2),
    )
    image_size = args.size #128
    image_observation_space = gym.spaces.Box(
        low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
    )
    obs_observation_space = gym.spaces.Box(
        low=-1, high=1, shape=(2,), dtype=np.float32
    )
    observation_space = gym.spaces.Dict({
            'state': gt_observation_space,
            'obs_state': obs_observation_space,
            'image': image_observation_space
        })
    print("Action Space", action_space)
    args.num_actions = action_space.n if hasattr(action_space, "n") else action_space.shape[0]

    
    expert_eps = collections.OrderedDict()
    tools.fill_expert_dataset_dubins(args, expert_eps)
    expert_dataset = make_dataset(expert_eps, args)
    # validation replay buffer
    expert_val_eps = collections.OrderedDict()
    tools.fill_expert_dataset_dubins(args, expert_val_eps, is_val_set=True)
    eval_dataset = make_dataset(expert_eps, args)

    print("Length of training data:", len(expert_eps))
    print("Length of validation data:", len(expert_val_eps))

    logdir = pathlib.Path(args.logdir).expanduser()
    logger = tools.Logger(logdir, 0)

    agent = Dreamer(
        observation_space,
        action_space,
        args,
        logger,
        expert_dataset,
    ).to(args.device)

    for step in trange(
            args.train_steps,
            desc="Training the RSSM",
            ncols=0,
            leave=False,
        ):        

            if (
                ((step + 1) % args.eval_every) == 0
                or step == 1
            ):
                #score, success = evaluate(
                #    other_dataset=expert_dataset, eval_prefix="pretrain"
                #)
                #lx_plot, tp, fn, fp, tn = agent.get_eval_plot()

                #logger.image("pretrain/lx_plot", np.transpose(lx_plot, (2, 0, 1)))
                
                #best_pretrain_success = tools.save_checkpoint(
                #    ckpt_name, step, success, best_pretrain_success, agent, logdir
                #)
                pass
    
            exp_data = next(expert_dataset)
            agent.pretrain_model_only(exp_data)