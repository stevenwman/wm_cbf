# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from io import BytesIO
from PIL import Image
import functools

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)
dreamer = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer)
sys.path.append(str(pathlib.Path(__file__).parent))
from dubin_multiobs_render import state_to_image_pil_hq
from generate_data_traj_cont_gap import failure_check_batch

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
import matplotlib.pyplot as plt

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
    eval_episode_num: int = 10
    reset_every: int =  0
    device: str = 'cuda:0'
    compile: bool = True
    precision: int =  32
    debug: bool =  False
    video_pred_log: bool =  True
    precision: int = 32
    action_repeat: int = 1
    steps: int = 10_000_000

    eval_every: int = 500
    log_every: int = 2000
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
    relu_weight: float = 100
    gp_weight: float = 10
    zero_sum_weight: float = 0.01

    x_min: float = -1.5
    x_max: float = 1.5
    y_min: float = -1.5
    y_max: float = 1.5
    size: List[int] = field(default_factory=lambda: [128, 128])
    speed: float = 1.
    turnRate: float = 1.25
    x_min: float = -1.5
    x_max: float = 1.5
    y_min: float = -1.5
    y_max: float = 1.5
    buffer: float = 0.1
    dt: float = 0.05
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
    # dataset_path: str = '/home/kensuke/WM_CBF/wm_cbf/wm_demos128_gap.pkl'
    dataset_path: str = '/home/clown2/Desktop/Work/Research/wm_cbf/wm_demos128_gap.pkl'
    num_trajs: int = 4000 # 2000 in paper
    num_train_trajs: int = 3800

    discount: int = 1.
    nx: int = 31
    ny: int = 31
    nz: int = 3
    obs_x: List[float] = field(default_factory=lambda: [0, 0])
    obs_y: List[float] = field(default_factory=lambda: [0.65, -0.65])
    obs_r: List[float] = field(default_factory=lambda: [0.5, 0.5])


def make_dataset(episodes, args):
    generator = tools.sample_episodes(episodes, args.batch_length)
    dataset = tools.from_generator(generator, args.batch_size)
    return dataset
    

class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, args, logger, dataset, expert_dataset=None):
        super(Dreamer, self).__init__()
        self._logger = logger
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
        self.fill_cache()

        model_params = {
                "params": list(self._wm.encoder.parameters())
                + list(self._wm.dynamics.parameters())
                + list(self._wm.heads["decoder"].parameters())
            }
        self.pretrain_params = list(model_params["params"])


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
                kl_free = self._args.kl_free
                dyn_scale = self._args.dyn_scale
                rep_scale = self._args.rep_scale
                # note: kl_loss is already sum of dyn_loss and rep_loss
                kl_loss, kl_value, dyn_loss, rep_loss = wm.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape

                losses = {}
                feat = wm.dynamics.get_feat(post)

                preds = {}
                for name, head in wm.heads.items():
                    # if name != "margin":
                    if 'margin' not in name:
                        grad_head = name in self._args.grad_heads
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
                # metrics = self.pretrain_opt(
                #     torch.mean(model_loss), self.pretrain_params
                # )
                metrics = wm._model_opt(
                    # TODO: look at where params are
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
        

        feat_detached = wm.dynamics.get_feat(post).detach()
        safe_data = torch.where(data["failure"] == 0.)
        unsafe_data = torch.where(data["failure"] == 1.)
        safe_dataset = feat_detached[safe_data]
        unsafe_dataset = feat_detached[unsafe_data]

        # gradient penalty head
        with tools.RequiresGrad(wm.heads["margin_gp"]):
            with torch.amp.autocast("cuda", enabled=wm._use_amp):
                pos = wm.heads["margin_gp"](safe_dataset)
                neg = wm.heads["margin_gp"](unsafe_dataset)
                gamma = self._args.gamma_lx
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
                    disc_interpolates = wm.heads["margin_gp"](interpolates)

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
                    relu_loss += torch.relu(gamma - pos).sum()
                if neg.numel() >0:
                    neg_mean = neg.mean()
                    zero_sum_loss += neg_mean
                    relu_loss += torch.relu(gamma + neg).sum()
                # print(neg.numel(), pos.numel())
                relu_weight = args.relu_weight
                gp_weight = args.gp_weight
                zero_sum_weight = args.zero_sum_weight
                loss = zero_sum_weight * zero_sum_loss + relu_weight * relu_loss + gp_weight * gp_loss
                
                metrics.update(wm.margin_gp_opt(loss, wm.heads["margin_gp"].parameters()))
                metrics["margin_gp"] = gp_loss.item()
                metrics["sign_loss"] = to_np(relu_loss)
                metrics["gp_loss"] = to_np(gp_loss)

        # no gradient penalty head
        with tools.RequiresGrad(wm.heads["margin_nogp"]):
            with torch.amp.autocast("cuda", enabled=wm._use_amp):
                pos = wm.heads["margin_nogp"](safe_dataset)
                neg = wm.heads["margin_nogp"](unsafe_dataset)
                # gamma = self._args.gamma_lx
                gamma = 0.75
                lx_loss = 0.0
                if pos.numel() > 0:
                    lx_loss += torch.relu(gamma - pos).mean()
                if neg.numel() > 0:
                    lx_loss += torch.relu(gamma + neg).mean()

                metrics["margin_nogp"] = lx_loss.item()
                metrics.update(wm.margin_nogp_opt(lx_loss, wm.heads["margin_nogp"].parameters()))
    
        metrics = {
            f"model_only_pretrain/{k}": v for k, v in metrics.items()
        }  # Add prefix model_pretrain to all metrics
        self._update_running_metrics(metrics)
        self._maybe_log_metrics()
        self._step += 1
        self._logger.step = self._step

    def _maybe_log_metrics(self, video_pred_log=False):
        if self._logger is not None:
            logged = False
            if self._should_log(self._step):
                for name, values in self._metrics.items():
                    if not np.isnan(np.mean(values)):
                        self._logger.scalar(name, float(np.mean(values)))
                        self._metrics[name] = []
                logged = True

            if video_pred_log and self._should_log_video(self._step):
                video_pred, video_pred2 = self._wm.video_pred(next(self._dataset))
                self._logger.video("train_openl_agent", to_np(video_pred))
                self._logger.video("train_openl_hand", to_np(video_pred2))
                logged = True

            if logged:
                self._logger.write(fps=True)

    def _update_running_metrics(self, metrics):
        for name, value in metrics.items():
            if name not in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)
    
    def fill_cache(self):
        print('filling cache')
        nx, ny, nz = self._args.nx, self._args.ny, self._args.nz
        self.nz =nz
        self.v = np.zeros((nx, ny, nz))
        v = self.v
        xs = np.linspace(self._args.x_min, self._args.x_max, nx)
        ys = np.linspace(self._args.y_min, self._args.y_max, ny)
        thetas= np.linspace(0, 2*np.pi, nz, endpoint=True)
        it = np.nditer(v, flags=['multi_index'])
        idxs = []  
        imgs = []
        labels = []
        it = np.nditer(v, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            x = xs[idx[0]]
            y = ys[idx[1]]
            theta = thetas[idx[2]]
            states = torch.tensor([x, y, theta])

            x_ob = torch.tensor(self._args.obs_x)
            y_ob = torch.tensor(self._args.obs_y)
            r_ob = torch.tensor(self._args.obs_r)

            if torch.any(failure_check_batch(states, x_ob, y_ob, r_ob)).item():
                labels.append(1) # unsafe
            else:
                labels.append(0) # safe
            # x = x - np.cos(theta)*1*0.05
            # y = y - np.sin(theta)*1*0.05
            x = x - self._args.dt * self._args.speed * np.cos(theta)
            y = y - self._args.dt * self._args.speed * np.sin(theta)

            imgs.append(state_to_image_pil_hq(np.array([x, y, theta]), self._args))
            idxs.append(idx)        
            it.iternext()
        idxs = np.array(idxs)
        self.idxs=idxs
        self.safe_idxs = np.where(np.array(labels) == 0)
        self.unsafe_idxs = np.where(np.array(labels) == 1)
        self.theta_lin = thetas[idxs[:,2]]
        self.imgs = imgs
        print('done!')

    def get_latent(self, thetas, imgs):

        states = np.expand_dims(np.expand_dims(thetas,1),1)
        imgs = np.expand_dims(imgs, 1)
        dummy_acs = np.zeros((np.shape(thetas)[0], 1))
        dummy_acs[np.arange(np.shape(thetas)[0]), :] = 0.
        firsts = np.ones((np.shape(thetas)[0], 1))
        lasts = np.zeros((np.shape(thetas)[0], 1))
        
        cos = np.cos(states)
        sin = np.sin(states)
        states = np.concatenate([cos, sin], axis=-1)
        data = {'obs_state': states, 'image': imgs, 'action': dummy_acs, 'is_first': firsts, 'is_terminal': lasts}

        data = self._wm.preprocess(data)
        embed = self._wm.encoder(data)

        post, prior = self._wm.dynamics.observe(
            embed, data["action"], data["is_first"]
            )
        feat = self._wm.dynamics.get_feat(post).detach()
        with torch.no_grad():  # Disable gradient calculation
            g_x = self._wm.heads["margin_gp"](feat).detach().cpu().numpy().squeeze()
            g_x_gp = self._wm.heads["margin_nogp"](feat).detach().cpu().numpy().squeeze()
        feat = self._wm.dynamics.get_feat(post).detach().cpu().numpy().squeeze()

        return g_x, g_x_gp, feat, post
    
    
    def get_eval_plot(self):
        self.eval()
        v = self.v.copy()
        v_gp = self.v.copy()
        
        g_x = []
        g_x_gp = []
        
        g_xlist, g_xgplist, _, _ = self.get_latent(self.theta_lin, self.imgs)
        g_x = g_x + g_xlist.tolist()
        g_x_gp = g_x_gp + g_xgplist.tolist()

        g_x = np.array(g_x)
        g_x_gp = np.array(g_x_gp)

        v[self.idxs[:, 0], self.idxs[:, 1], self.idxs[:, 2]] = g_x
        v_gp[self.idxs[:, 0], self.idxs[:, 1], self.idxs[:, 2]] = g_x_gp

        tp  = np.where(g_x[self.safe_idxs] > 0)
        fn  = np.where(g_x[self.safe_idxs] <= 0)
        fp  = np.where(g_x[self.unsafe_idxs] > 0)
        tn  = np.where(g_x[self.unsafe_idxs] <= 0)

        tp_gp  = np.where(g_x_gp[self.safe_idxs] > 0)
        fn_gp  = np.where(g_x_gp[self.safe_idxs] <= 0)
        fp_gp  = np.where(g_x_gp[self.unsafe_idxs] > 0)
        tn_gp  = np.where(g_x_gp[self.unsafe_idxs] <= 0)

        vmax = 2
        vmin = -2
        fig, axes = plt.subplots(self.nz, 2, figsize=(12, self.nz*6))
        fig_gp, axes_gp = plt.subplots(self.nz, 2, figsize=(12, self.nz*6))

        for i in range(self.nz):
            ax = axes[i, 0]
            im = ax.imshow(
                v[:, :, i].T, interpolation='none', extent=np.array([
                self._args.x_min, self._args.x_max, self._args.y_min, self._args.y_max, ]), origin="lower",
                cmap="seismic", vmin=vmin, vmax=vmax, zorder=-1
            )
            cbar = fig.colorbar(
                im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
            )
            cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
            ax.set_title(r'$l(x)$', fontsize=18)

            ax_gp = axes_gp[i, 0]
            im_gp = ax_gp.imshow(
                v_gp[:, :, i].T, interpolation='none', extent=np.array([
                self._args.x_min, self._args.x_max, self._args.y_min, self._args.y_max, ]), origin="lower",
                cmap="seismic", vmin=vmin, vmax=vmax, zorder=-1
            )
            cbar_gp = fig_gp.colorbar(
                im_gp, ax=ax_gp, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
            )
            cbar_gp.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
            
            ax_gp.set_title(r'$l(x) gp$', fontsize=18)

            ax = axes[i, 1]
            im = ax.imshow(
                v[:, :, i].T > 0, interpolation='none', extent=np.array([
                self._args.x_min, self._args.x_max, self._args.y_min, self._args.y_max, ]), origin="lower",
                cmap="seismic", vmin=-1, vmax=1, zorder=-1
            )
            cbar = fig.colorbar(
                im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
            )
            cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
            ax.set_title(r'$l(x) > 0$', fontsize=18)
            fig.tight_layout()

            ax_gp = axes_gp[i, 1]
            im_gp = ax_gp.imshow(
                v_gp[:, :, i].T > 0, interpolation='none', extent=np.array([
                self._args.x_min, self._args.x_max, self._args.y_min, self._args.y_max, ]), origin="lower",
                cmap="seismic", vmin=-1, vmax=1, zorder=-1
            )
            cbar_gp = fig_gp.colorbar(
                im_gp, ax=ax_gp, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
            )
            cbar_gp.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
            ax_gp.set_title(r'$l(x) > 0$', fontsize=18)
            fig_gp.tight_layout()

            xs = self._args.obs_x
            ys = self._args.obs_y
            rs = self._args.obs_r
            
            for ob in range(len(xs)):
                # circle = plt.Circle((0, 0), self._args.obs_r, fill=False, color='blue', label = 'GT boundary')
                circle = plt.Circle((xs[ob], ys[ob]), rs[ob], fill=False, color='blue', label = 'GT boundary')
                # Add the circle to the plot
                axes[i,0].add_patch(circle)
                circle = plt.Circle((xs[ob], ys[ob]), rs[ob], fill=False, color='blue', label = 'GT boundary')
                axes_gp[i,0].add_patch(circle)

                # circle2 = plt.Circle((0, 0), self._args.obs_r, fill=False, color='blue', label = 'GT boundary')
                circle2 = plt.Circle((xs[ob], ys[ob]), rs[ob], fill=False, color='blue', label = 'GT boundary')
                axes[i,1].add_patch(circle2)
                circle2 = plt.Circle((xs[ob], ys[ob]), rs[ob], fill=False, color='blue', label = 'GT boundary')
                axes_gp[i,1].add_patch(circle2)

            axes[i,0].set_aspect('equal')
            axes[i,1].set_aspect('equal')
            axes_gp[i,0].set_aspect('equal')
            axes_gp[i,1].set_aspect('equal')

        fp_g = np.shape(fp)[1]
        fn_g = np.shape(fn)[1]
        tp_g = np.shape(tp)[1]
        tn_g = np.shape(tn)[1]
        tot = fp_g + fn_g + tp_g + tn_g
        fig.suptitle(r"$TP={:.0f}\%$ ".format(tp_g/tot * 100) + r"$TN={:.0f}\%$ ".format(tn_g/tot * 100) + r"$FP={:.0f}\%$ ".format(fp_g/tot * 100) +r"$FN={:.0f}\%$".format(fn_g/tot * 100),
            fontsize=10,)
        
        fp_g_gp = np.shape(fp_gp)[1]
        fn_g_gp = np.shape(fn_gp)[1]
        tp_g_gp = np.shape(tp_gp)[1]
        tn_g_gp = np.shape(tn_gp)[1]
        tot_gp = fp_g_gp + fn_g_gp + tp_g_gp + tn_g_gp
        fig_gp.suptitle(r"$TP={:.0f}\%$ ".format(tp_g_gp/tot_gp * 100) + r"$TN={:.0f}\%$ ".format(tn_g_gp/tot_gp * 100) + r"$FP={:.0f}\%$ ".format(fp_g_gp/tot_gp * 100) +r"$FN={:.0f}\%$".format(fn_g_gp/tot_gp * 100),
            fontsize=10,)
        
        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        plot = Image.open(buf).convert("RGB")
        buf.close()

        bufgp = BytesIO()
        fig_gp.savefig(bufgp, format="png")
        plt.close(fig_gp)
        bufgp.seek(0)
        plot_gp = Image.open(bufgp).convert("RGB")
        bufgp.close()
        
        self.train()
        return np.array(plot), tp, fn, fp, tn, np.array(plot_gp), tp_g_gp, fn_g_gp, fp_g_gp, tn_g_gp
    

if __name__ == "__main__":
    args = tyro.cli(Args)

    def evaluate(other_dataset=None, eval_prefix=""):
        agent.eval()
        
        eval_policy = functools.partial(agent, training=False)

        # For Logging (1 episode)
        if args.video_pred_log:
            video_pred = agent._wm.video_pred(next(eval_dataset))
            logger.video("eval_recon/openl_agent", to_np(video_pred))

            if other_dataset:
                video_pred = agent._wm.video_pred(next(other_dataset))
                logger.video("train_recon/openl_agent", to_np(video_pred))

        
        logger.write(step=logger.step)
        #recon_eval = eval_obs_recon()  # testing observation reconstruction

        agent.train()
        return 0, 0 #recon_eval, recon_eval



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
    image_size = args.size[0] #128
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

    logdir = pathlib.Path(args.logdir).expanduser()
    args.traindir = logdir / "train_eps"
    args.evaldir = logdir / "eval_eps"
    args.steps //= args.action_repeat
    args.eval_every //= args.action_repeat
    args.log_every //= args.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    args.traindir.mkdir(parents=True, exist_ok=True)
    args.evaldir.mkdir(parents=True, exist_ok=True)
    # step in logger is environmental step
    logger = tools.Logger(logdir, 0)

    agent = Dreamer(
        observation_space,
        action_space,
        args,
        logger,
        expert_dataset,
    ).to(args.device)
    ckpt_name = "rssm_ckpt" 
    best_pretrain_success = float("inf")
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
                lx_plot, tp, fn, fp, tn, lx_plot_gp, tp_g, fn_g, fp_g, tn_g = agent.get_eval_plot()

                logger.image("pretrain/lx_plot", np.transpose(lx_plot, (2, 0, 1)))
                logger.image("pretrain/lx_plot_gp", np.transpose(lx_plot_gp, (2, 0, 1)))

                score, success = evaluate(
                   other_dataset=expert_dataset, eval_prefix="pretrain"
                )
                best_pretrain_success = tools.save_checkpoint(
                    ckpt_name, step, 0, best_pretrain_success, agent, logdir
                )
    
            exp_data = next(expert_dataset)
            agent.pretrain_model_only(exp_data)