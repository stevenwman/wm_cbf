import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer)
sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import torch
from torch import nn
import collections

from tqdm import trange
from termcolor import cprint
import matplotlib.pyplot as plt
import gym
from io import BytesIO
from PIL import Image
from copy import deepcopy

to_np = lambda x: x.detach().cpu().numpy()
from generate_data_traj_cont_gap import failure_check_batch
from dubin_multiobs_render import state_to_image_pil_hq

class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset

        # 1. Instantiate the WorldModel first.
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)

        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter

        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)

        self._make_pretrain_opt()
        self.fill_cache()

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _make_pretrain_opt(self):
        config = self._config
        use_amp = True if config.precision == 16 else False
        if (
            config.rssm_train_steps > 0
            or config.from_ckpt is not None
        ):
            # have separate lrs/eps/clips for actor and model
            # https://pytorch.org/docs/master/optim.html#per-parameter-options
            standard_kwargs = {
                "lr": config.model_lr,
                "eps": config.opt_eps,
                "clip": config.grad_clip,
                "wd": config.weight_decay,
                "opt": config.opt,
                "use_amp": use_amp,
            }
            model_params = {
                "params": list(self._wm.encoder.parameters())
                + list(self._wm.dynamics.parameters())
                + list(self._wm.heads["decoder"].parameters())
                + list(self._wm.heads["cont"].parameters())
            }
            self.pretrain_params = list(model_params["params"])
            self.pretrain_opt = tools.Optimizer(
                "pretrain_opt", [model_params], **standard_kwargs
            )

    def _update_running_metrics(self, metrics):
        for name, value in metrics.items():
            if name not in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

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

                # In dreamer_offline.py, inside the Dreamer class

    def pretrain_model_only(self, data, step=None):
        metrics = {}
        wm = self._wm
        data = wm.preprocess(data)
        
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

                if (step <= self._config.rssm_train_steps):
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
                        if name == "cont":
                            cont_loss = -pred.log_prob(data[name])
                        else:
                            loss = -pred.log_prob(data[name])
                            assert loss.shape == embed.shape[:2], (name, loss.shape)
                            losses[name] = loss
                        
                    recon_loss = sum(losses.values())

                model_loss = kl_loss + recon_loss + cont_loss
                metrics = self.pretrain_opt(
                    torch.mean(model_loss), self.pretrain_params
                )

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_loss"] = to_np(kl_loss)
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl_value"] = to_np(torch.mean(kl_value))
        metrics["cont_loss"] = to_np(cont_loss)

        with torch.amp.autocast("cuda", enabled=wm._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(wm.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(wm.dynamics.get_dist(post).entropy())
            )

        with tools.RequiresGrad(wm.value_out):
            with torch.amp.autocast("cuda", enabled=wm._use_amp):
                # We already have `post` from the world model training above.
                # Re-calculating it is inefficient but fine. For simplicity, let's keep it.
                embed = wm.encoder(data)
                post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
                feat = wm.dynamics.get_feat(post).detach()
                
                reward_out = data['reward'].unsqueeze(-1)
                discount = wm._config.discount * data['cont']
                value_out = wm.value_out(feat).mode()

                # THIS IS THE CORRECT, MULTI-STEP TARGET CALCULATION
                target_out = tools.lambda_return(
                    reward_out[1:], value_out[:-1], discount[1:],
                    bootstrap=value_out[-1], lambda_=wm._config.discount_lambda, axis=0
                )

                target_out = torch.stack(target_out, dim=1)
                zero_reward_mask_out = (reward_out[:-1] == 0.0)

                # Where the mask is True, clamp the target to 0.
                # Otherwise, use the calculated lambda-return target.
                clamped_target_out = torch.where(zero_reward_mask_out, torch.zeros_like(target_out), target_out)
                value_out_pred = wm.value_out(feat[:-1])
                value_out_loss = -value_out_pred.log_prob(clamped_target_out.detach())

                if wm._config.critic['slow_target']:
                    slow_target_out = wm._slow_value_out(feat[:-1]).mode()
                    clamped_slow_target_out = torch.where(zero_reward_mask_out, torch.zeros_like(slow_target_out), slow_target_out)
                    value_out_loss -= value_out_pred.log_prob(clamped_slow_target_out.detach())

                weights = torch.cumprod(torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0).detach()
                value_out_loss = torch.mean(weights.squeeze(-1)[:-1] * value_out_loss)

            metrics.update(wm._value_out_opt(value_out_loss, wm.value_out.parameters()))

        with tools.RequiresGrad(wm.value_in):
            with torch.amp.autocast("cuda", enabled=wm._use_amp):
                # We already have `post` from the world model training above.
                # Re-calculating it is inefficient but fine. For simplicity, let's keep it.
                embed = wm.encoder(data)
                post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
                feat = wm.dynamics.get_feat(post).detach()
                
                reward_out = data['reward'].unsqueeze(-1)
                reward_in = reward_out + 1
                discount = wm._config.discount * data['cont']

                value_in = wm.value_in(feat).mode()
                target_in = tools.lambda_return(
                    reward_in[:-1], value_in[:-1], discount[:-1],
                    bootstrap=value_in[-1], lambda_=wm._config.discount_lambda, axis=0
                )

                target_in = torch.stack(target_in, dim=1)
                zero_reward_mask_in = (reward_in[:-1] == 0.0)

                # Where the mask is True, clamp the target to 0.
                # Otherwise, use the calculated lambda-return target.
                clamped_target_in = torch.where(zero_reward_mask_in, torch.zeros_like(target_in), target_in)
                value_in_pred = wm.value_in(feat[:-1])
                value_in_loss = -value_in_pred.log_prob(clamped_target_in.detach())

                if wm._config.critic['slow_target']:
                    slow_target_in = wm._slow_value_in(feat[:-1]).mode()
                    clamped_slow_target_in = torch.where(zero_reward_mask_in, torch.zeros_like(slow_target_in), slow_target_in)
                    value_in_loss -= value_in_pred.log_prob(clamped_slow_target_in.detach())

                weights = torch.cumprod(torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0).detach()
                value_in_loss = torch.mean(weights.squeeze(-1)[:-1] * value_in_loss)

            metrics.update(wm._value_in_opt(value_in_loss, wm.value_in.parameters()))

        metrics.update(tools.tensorstats(wm.value_out(feat).mode(), "value_out"))
        metrics.update(tools.tensorstats(wm.value_in(feat).mode(), "value_in"))
        metrics.update(tools.tensorstats(target_out, "target_out"))
        metrics.update(tools.tensorstats(target_in, "target_in"))
        metrics['offline_value_out_loss'] = to_np(value_out_loss)
        metrics['offline_value_in_loss'] = to_np(value_in_loss)

        metrics = {
            f"model_only_pretrain/{k}": v for k, v in metrics.items()
        }  # Add prefix model_pretrain to all metrics
        self._update_running_metrics(metrics)
        self._maybe_log_metrics()
        self._step += 1
        self._logger.step = self._step

    def pretrain_regress_obs(self, data, obs_mlp, obs_opt, eval=False):
        wm = self._wm
        # actor = self._task_behavior.actor
        data = wm.preprocess(data)
        if eval:
            obs_mlp.eval()
        with tools.RequiresGrad(obs_mlp):
            with torch.amp.autocast("cuda", enabled=wm._use_amp):
                embed = self._wm.encoder(data)
                post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])

                feat = self._wm.dynamics.get_feat(prior).detach() # want the imagined prior to be strong
                target = torch.Tensor(data["privileged_state"]).to(self._config.device)
                pred_state = obs_mlp(feat)
                obs_loss = torch.mean((pred_state - target) ** 2)
            if not eval:
                obs_opt(torch.mean(obs_loss), obs_mlp.parameters())
            else:
                obs_mlp.train()
        return obs_loss.item()
    
    def fill_cache(self):
        print('filling cache')
        nx, ny, nz = self._config.nx, self._config.ny, self._config.nz
        self.nz =nz
        self.v = np.zeros((nx, ny, nz))
        v = self.v
        xs = np.linspace(self._config.x_min, self._config.x_max, nx)
        ys = np.linspace(self._config.y_min, self._config.y_max, ny)
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

            x_ob = torch.tensor(self._config.obs_x)
            y_ob = torch.tensor(self._config.obs_y)
            r_ob = torch.tensor(self._config.obs_r)

            if torch.any(failure_check_batch(states, x_ob, y_ob, r_ob)).item():
                labels.append(1) # unsafe
            else:
                labels.append(0) # safe
            x = x - self._config.dt * self._config.speed * np.cos(theta)
            y = y - self._config.dt * self._config.speed * np.sin(theta)

            imgs.append(state_to_image_pil_hq(np.array([x, y, theta]), self._config))
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
            value_out_pred = self._wm.value_out(feat).mode().detach().cpu().numpy().squeeze()
            value_in_pred = self._wm.value_in(feat).mode().detach().cpu().numpy().squeeze()
        feat = feat.cpu().numpy().squeeze()

        return value_out_pred, value_in_pred, feat, post # Return the value prediction instead of g_x

    def get_eval_plot(self):
        self.eval()
        v_out = deepcopy(self.v)
        v_in = deepcopy(self.v)

        value_out_preds, value_in_preds, _, _ = self.get_latent(self.theta_lin, self.imgs)
        v_out[self.idxs[:, 0], self.idxs[:, 1], self.idxs[:, 2]] = value_out_preds
        v_in[self.idxs[:, 0], self.idxs[:, 1], self.idxs[:, 2]] = value_in_preds

        v_tot = -(v_out + v_in)

        vmax = round(max(np.max(v_out), np.max(v_in), np.max(v_tot), 0),1)
        vmin = round(min(np.min(v_out), np.min(v_in), np.min(v_tot), -vmax),1)
        vmax = min(vmax, 2.)
        vmin = max(vmin, -2.)
        fig, axes = plt.subplots(self.nz, 4, figsize=(4 * 6, self.nz*6))

        for i in range(self.nz):
            theta_idx = np.floor(i / self.nz * v_tot.shape[2]).astype(int)
            theta = i / self.nz * 360

            ax = axes[i, 0]
            im = ax.imshow(
                v_out[:, :, theta_idx].T, interpolation='none', extent=np.array([
                self._config.x_min, self._config.x_max, self._config.y_min, self._config.y_max, ]), origin="lower",
                cmap="seismic", vmin=vmin, vmax=vmax, zorder=-1
            )
            cbar = fig.colorbar(
                im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
            )
            cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
            # ax.set_title(r'$g(x)$', fontsize=18)
            ax.set_title(rf'Expected Value $V out(s)$ at {theta:.2f} deg', fontsize=10)

            ax = axes[i, 1]
            im = ax.imshow(
                v_in[:, :, theta_idx].T, interpolation='none', extent=np.array([
                self._config.x_min, self._config.x_max, self._config.y_min, self._config.y_max, ]), origin="lower",
                cmap="seismic", vmin=-1, vmax=1, zorder=-1
            )
            cbar = fig.colorbar(
                im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
            )
            cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
            ax.set_title(rf'Expected Value $V in(s)$ at {theta:.2f} deg', fontsize=10)

            ax = axes[i, 2]
            im = ax.imshow(
                v_tot[:, :, theta_idx].T, interpolation='none', extent=np.array([
                self._config.x_min, self._config.x_max, self._config.y_min, self._config.y_max, ]), origin="lower",
                cmap="seismic", vmin=-1, vmax=1, zorder=-1
            )
            cbar = fig.colorbar(
                im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
            )
            cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
            ax.set_title(rf'Expected Value $V tot(s)$ at {theta:.2f} deg', fontsize=10)

            ax = axes[i, 3]
            im = ax.imshow(
                v_tot[:, :, i].T > 0, interpolation='none', extent=np.array([
                self._config.x_min, self._config.x_max, self._config.y_min, self._config.y_max, ]), origin="lower",
                cmap="seismic", vmin=-1, vmax=1, zorder=-1
            )
            cbar = fig.colorbar(
                im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
            )
            cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
            ax.set_title(rf'Expected Value $V_tot(s) > 0$ at {theta:.2f} deg', fontsize=10)


            fig.tight_layout()

            xs = self._config.obs_x
            ys = self._config.obs_y
            rs = self._config.obs_r
            for ob in range(len(xs)):
                circle = plt.Circle((xs[ob], ys[ob]), rs[ob], fill=False, color='blue', label = 'GT boundary')
                # Add the circle to the plot
                axes[i,0].add_patch(circle)

                circle2 = plt.Circle((xs[ob], ys[ob]), rs[ob], fill=False, color='blue', label = 'GT boundary')
                axes[i,1].add_patch(circle2)

                circle3 = plt.Circle((xs[ob], ys[ob]), rs[ob], fill=False, color='blue', label = 'GT boundary')
                # Add the circle to the plot
                axes[i,2].add_patch(circle3)

                circle4 = plt.Circle((xs[ob], ys[ob]), rs[ob], fill=False, color='blue', label = 'GT boundary')
                axes[i,3].add_patch(circle4)

            axes[i,0].set_aspect('equal')
            axes[i,1].set_aspect('equal')
            axes[i,2].set_aspect('equal')
            axes[i,3].set_aspect('equal')

        buf = BytesIO()

        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        plot = Image.open(buf).convert("RGB")
        self.train()
        return np.array(plot), [], [], [], []  # Placeholder values for tp, fn, fp, tn

def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    # logger = tools.DummyLogger(logdir, config.action_repeat * step)

    print("Create envs.")
    
    action_space = gym.spaces.Box(
        low=-config.turnRate, high=config.turnRate, shape=(1,), dtype=np.float32
    )
    bounds = np.array([[config.x_min, config.x_max], [config.y_min, config.y_max], [0, 2 * np.pi]])
    low = bounds[:, 0]
    high = bounds[:, 1]
    midpoint = (low + high) / 2.0
    interval = high - low
    gt_observation_space = gym.spaces.Box(
        np.float32(midpoint - interval/2),
        np.float32(midpoint + interval/2),
    )
    image_size = config.size[0] #128
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
    config.num_actions = action_space.n if hasattr(action_space, "n") else action_space.shape[0]

    
    expert_eps = collections.OrderedDict()
    print(expert_eps)
    tools.fill_expert_dataset_dubins(config, expert_eps)
    expert_dataset = make_dataset(expert_eps, config)
    # validation replay buffer
    expert_val_eps = collections.OrderedDict()
    tools.fill_expert_dataset_dubins(config, expert_val_eps, is_val_set=True)
    eval_dataset = make_dataset(expert_eps, config)

    print("Length of training data:", len(expert_eps))
    print("Length of validation data:", len(expert_val_eps))

    print("Simulate agent.")
    agent = Dreamer(
        observation_space,
        action_space,
        config,
        logger,
        expert_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    init_step = 0

    def log_plot(title, data):
        buf = BytesIO()
        plt.plot(np.arange(len(data)), data)
        plt.title(title)
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        plot = Image.open(buf).convert("RGB")
        plot_arr = np.array(plot)
        logger.image("pretrain/" + title, np.transpose(plot_arr, (2, 0, 1)))
    
    def evaluate(other_dataset=None, eval_prefix=""):
        agent.eval()
        
        eval_policy = functools.partial(agent, training=False)

        # For Logging (1 episode)
        if config.video_pred_log:
            video_pred = agent._wm.video_pred(next(eval_dataset))
            logger.video("eval_recon/openl_agent", to_np(video_pred))

            if other_dataset:
                video_pred = agent._wm.video_pred(next(other_dataset))
                logger.video("train_recon/openl_agent", to_np(video_pred))

        
        logger.write(step=logger.step)
        #recon_eval = eval_obs_recon()  # testing observation reconstruction

        agent.train()
        return 0, 0 #recon_eval, recon_eval
    # ==================== Pretrain ====================
    total_train_steps = config.rssm_train_steps 
    print(total_train_steps)
    if total_train_steps > 0:
        
        cprint(
            f"Pretraining for {total_train_steps=}",
            color="cyan",
            attrs=["bold"],
        )
        ckpt_name = "rssm_ckpt" 
        best_pretrain_success = float("inf")
        for step in trange(
            total_train_steps-init_step,
            desc="Training the RSSM",
            ncols=0,
            leave=False,
        ):
            if (
                ((step +init_step + 1) % config.eval_every) == 0
                or step+init_step == 1
            ):
                
                lx_plot, tp, fn, fp, tn = agent.get_eval_plot()

                logger.image("pretrain/lx_plot", np.transpose(lx_plot, (2, 0, 1)))
                
                score, success = evaluate(
                    other_dataset=expert_dataset, eval_prefix="pretrain"
                )
                best_pretrain_success = tools.save_checkpoint(
                    ckpt_name, step+init_step, success, best_pretrain_success, agent, logdir
                )

    
            exp_data = next(expert_dataset)
            agent.pretrain_model_only(exp_data, step+init_step)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()

    yaml = yaml.YAML(typ="safe", pure=True)
    configs = yaml.load(
        (pathlib.Path(sys.argv[0]).parent / "../configs_gap_crv.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
