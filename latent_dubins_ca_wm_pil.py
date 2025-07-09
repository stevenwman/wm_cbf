import numpy as np
import torch
import ruamel.yaml as yaml
from PytorchReachability.PyHJ.data import Batch
from PytorchReachability.PyHJ.exploration import GaussianNoise
from PytorchReachability.PyHJ.utils.net.common import Net
from PytorchReachability.PyHJ.utils.net.continuous import Actor, Critic
from PytorchReachability.PyHJ.policy import avoid_DDPGPolicy_annealing as DDPGPolicy
import pathlib, argparse, sys, gym, os
from dubin_render import state_to_image_pil_hq
cwd = os.getcwd()
print(cwd)

dreamer_dir = 'PytorchReachability/dreamerv3-torch'
ckpt_path = 'rssm_ckpt.pt'
policy_path = 'policy.pth'
config_path = 'PytorchReachability/configs.yaml'

sys.path.append(os.path.abspath(dreamer_dir))
import models, tools

def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value

def get_args(cfg_path=config_path):
    yml = yaml.YAML(typ="safe", pure=True)
    configs = yml.load(pathlib.Path(f"{cwd}/{cfg_path}").read_text())
    name_list = ["defaults"]
    defaults = {}

    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    final_config = parser.parse_args([])
    return final_config

class LatentDubinContinuousAction:
    """
    Class to calculate the latent Dubin Continuous Action value function.
    """

    def __init__(self):
        """
        Init function
        """
        self.args = get_args()
        self.np_expdim = lambda x: np.expand_dims(x, axis=0)
        self.unsqz = lambda x, ax: np.expand_dims(x, axis=ax)
        self.dbl_unsqz = lambda x, ax: np.expand_dims(np.expand_dims(x, axis=ax), axis=ax)
        self._init_wm()

    def dyn_step_back(self, s0: torch.Tensor) -> torch.Tensor:
        """
        Take a single timestep backward. Because the WM requires 
        a single forward step to compute latent and features.
        """
        v = self.args.speed
        dt = self.args.dt
        bs = s0.shape[0]
        s, s_prev = torch.zeros(bs, 3), torch.zeros(bs, 3)
        s0 = s0.clone().detach()
        s[:,0], s[:,1], s[:,2] = s0[:,0], s0[:,1], s0[:,2]
        s_prev[:,0] = s[:,0] - v*dt*torch.cos(s[:,2])
        s_prev[:,1] = s[:,1] - v*dt*torch.sin(s[:,2])
        s_prev[:,2] = s[:,2]
        return s_prev
    
    def state_to_data(self, s0: torch.Tensor) -> dict:
        """
        Convert the state to data for the latent Dubin environment.
        """
        state_obs, img_obs, state_gt, dones, acs = ([] for _ in range(5))
        
        for i in range(s0.shape[0]):
            print(f"Processing state {i} of {s0.shape[0]}", end='\r')
            s = s0[i]
            ac = 0 * torch.rand(1)
            state_obs.append(s[2].numpy()) # get to observe theta
            state_gt.append(s.numpy()) # gt state
            dones.append(1)
            acs.append(ac)

            img_array = state_to_image_pil_hq(s, self.args)

            img_obs.append(img_array)

        demo = {}
        demo['obs'] = {'image': img_obs, 'state': state_obs, 'priv_state': state_gt}
        demo['actions'] = acs
        demo['dones'] = dones

        return demo

    def demo_to_traj(self, demos: dict) -> dict:       
        """
        Convert demo dictionary to trajectory ready to be pre-processed
        """

        traj = demos
        traj_to_pp = {}
        length = len(traj['obs']['state'])

        traj_to_pp['image'] = self.unsqz(np.array(traj["obs"]['image']), ax=1)

        traj_to_pp["state"] = self.dbl_unsqz(np.array(traj["obs"]['state']), ax=-1)
        traj_to_pp["privileged_state"] = self.unsqz(np.array(traj['obs']["priv_state"], dtype=np.float32), ax=1)
        traj_to_pp['obs_state'] = self.unsqz(np.array([np.cos(traj['obs']['state']),
                                                       np.sin(traj['obs']['state'])], dtype=np.float32).T, ax=1)
        traj_to_pp["reward"] = np.zeros((length, 1), dtype=np.float32)

        traj_to_pp["is_first"] = np.zeros((length, 1), dtype=np.float32)
        traj_to_pp["is_first"][0,0] = 1
        traj_to_pp["is_first"] = traj_to_pp["is_first"] != 0

        traj_to_pp["is_last"] = self.unsqz(np.array(traj["dones"], dtype=np.bool_), ax=-1)
        traj_to_pp["is_terminal"] = traj_to_pp["is_last"]
        traj_to_pp["discount"] = np.ones((length, 1), dtype=np.float32)
        traj_to_pp["action"] = self.dbl_unsqz(np.array(traj["actions"], dtype=np.float32), ax=-1)
        
        return traj_to_pp
    
    def find_a(self, state):
        """
        Use the policy to find the safe action for a given state.
        """
        tmp_obs = np.array(state).reshape(state.shape[0],state.shape[-1])
        tmp_batch = Batch(obs = tmp_obs, info = Batch())
        tmp = self.policy(tmp_batch, model = "actor_old").act
        act = self.policy.map_action(tmp).cpu().detach().numpy().flatten()
        return act

    def evaluate_V(self, state):
        """
        Use the safe value function to find the value for a given state.
        """
        tmp_obs = np.array(state).reshape(state.shape[0],state.shape[-1])
        tmp_batch = Batch(obs = tmp_obs, info = Batch())
        tmp = self.policy.critic(tmp_batch.obs, self.policy(tmp_batch, model="actor_old").act)
        return tmp.cpu().detach().numpy().flatten()
    
    def _init_wm(self):
        """
        Initialize world model and value function.
        """
        args=self.args
        config=args

        # set up the environment spaces
        image_size = config.size[0] #128
        img_obs_space = gym.spaces.Box(low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8)
        obs_space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        # hardcoded env bounds
        high = np.array([config.x_max, config.y_max, 2*np.pi,])
        low = np.array([config.x_min, config.y_min, 0.,])
        gt_observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        observation_space = gym.spaces.Dict({
            'obs_state': obs_space,
            'image': img_obs_space,
            'state': gt_observation_space,
            })
        u_max = config.turnRate
        action_space = gym.spaces.Box(low=-u_max, high=u_max, shape=(1,), dtype=np.float32)
        config.num_actions = action_space.n if hasattr(action_space, "n") else action_space.shape[0]

        # load 
        config.eval_state_mean = True
        wm = models.WorldModel(observation_space, action_space, 0, config)
        wm.to(config.device)
        checkpoint = torch.load(ckpt_path, weights_only=True)
        state_dict = {k[14:]:v for k,v in checkpoint['agent_state_dict'].items() if '_wm' in k}
        wm.load_state_dict(state_dict)
        
        # hardcoded latent dim
        args.state_shape = (1,1,544,)
        args.action_shape = (1,)
        args.max_action = u_max

        # seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        activation_map = {
            'ReLU': torch.nn.ReLU,
            'Tanh': torch.nn.Tanh,
            'Sigmoid': torch.nn.Sigmoid,
            'SiLU': torch.nn.SiLU
        }
        actor_activation = activation_map.get(args.actor_activation)
        critic_activation = activation_map.get(args.critic_activation)

        assert args.critic_net is not None, "Please provide critic_net!"
        critic_net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.critic_net,
            activation=critic_activation,
            concat=True,
            device=args.device
            )

        critic = Critic(critic_net, device=args.device).to(args.device)
        critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
        
        actor_net = Net(args.state_shape, 
                        hidden_sizes=args.control_net,
                        activation=actor_activation, 
                        device=args.device)
        actor = Actor(actor_net, 
                    args.action_shape, 
                    max_action=args.max_action, 
                    device=args.device).to(args.device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

        policy = DDPGPolicy(
            critic,
            critic_optim,
            tau=args.tau,
            gamma=args.gamma_pyhj,
            exploration_noise=GaussianNoise(sigma=args.exploration_noise),
            reward_normalization=args.rew_norm,
            estimation_step=args.n_step,
            action_space=action_space,
            actor=actor,
            actor_optim=actor_optim,
            actor_gradient_steps=args.actor_gradient_steps,
            )

        # load policy
        policy.load_state_dict(torch.load(policy_path, weights_only=True))
        self.wm = wm
        self.policy = policy
        self.wm.eval()
        self.policy.eval()

    def priv_state_to_V(self, state):
        s_curr = state
        s_prev = self.dyn_step_back(s_curr)
        data_pts = self.state_to_data(s_prev)
        traj = self.demo_to_traj(data_pts)

        bs = traj['state'].shape[0] # batch size
        action = torch.zeros((bs,1,1), device='cuda:0')
        is_first = torch.ones((bs,1), device='cuda:0')

        proc_data = self.wm.preprocess(traj)
        latent,_ = self.wm.dynamics.observe(self.wm.encoder(proc_data), action, is_first)
        latent['stoch'] = latent['mean']
        for k, v in latent.items(): latent[k] = v[:, [-1]]
        feat = self.wm.dynamics.get_feat(latent).detach().cpu().numpy() 
        value = self.evaluate_V(feat)

        # act = self.find_a(feat)
        # pr_state = proc_data['privileged_state'][0,0].cpu()
        # print("privileged state: ", pr_state)
        # print("safe action: ", act)
        # print("safe value: ", value)
        return value

    def obs_to_brt(self, data_pts):
        """
        Convert the observation (image and state) to a state and then to 
        a value and least-restrictive action.
        """
        traj = self.demo_to_traj(data_pts)

        bs = traj['state'].shape[0] # batch size
        action = torch.zeros((bs,1,1), device='cuda:0')
        is_first = torch.ones((bs,1), device='cuda:0')

        proc_data = self.wm.preprocess(traj)
        latent,_ = self.wm.dynamics.observe(self.wm.encoder(proc_data), action, is_first)
        latent['stoch'] = latent['mean']
        for k, v in latent.items(): latent[k] = v[:, [-1]]
        feat = self.wm.dynamics.get_feat(latent).detach().cpu().numpy() 
        value = self.evaluate_V(feat)
        lr_act = self.find_a(feat)

        return value, lr_act
    

