import argparse
import numpy as np
import torch
import pickle
import pathlib
import ruamel.yaml as yaml
import os, sys
cwd = os.getcwd() # make sure you're in repo root directory 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer_dir)
sys.path.append(cwd)
import tools
from dubin_multiobs_render import state_to_image_pil_hq

config_path = 'PytorchReachability/configs_gap.yaml'


def failure_check_batch(state: torch.tensor, xs: torch.tensor, ys: torch.tensor, rs: torch.tensor) -> torch.tensor:
	"""
	Check if the states are in the failure set defined by the circles with centers (xs, ys) and radii rs.
	"""
	return torch.norm(state[0:2].unsqueeze(1) - torch.stack((xs, ys)).unsqueeze(0), dim=1) < rs.unsqueeze(1).T

def get_init_state(config):    
	# don't sample inside the failure set
	states = None
	# while np.linalg.norm(states[:2] - np.array([config.obs_x, config.obs_y])) < config.obs_r:
	xs = torch.tensor(config.obs_x)
	ys = torch.tensor(config.obs_y)
	rs = torch.tensor(config.obs_r)

	while states is None or torch.any(failure_check_batch(states, xs, ys, rs)):
		states = torch.rand(3)
		states[0] *= (config.x_max-config.buffer) - (config.x_min + config.buffer)
		states[1] *= (config.y_max-config.buffer) - (config.y_min + config.buffer)
		states[0] += config.x_min + config.buffer
		states[1] += config.y_min + config.buffer

	# so that the trajectory doesn't immediately go out of bounds
	states[2] = torch.atan2(-states[1], -states[0]) + np.random.normal(0, 1)
	states[2] = states[2] % (2*np.pi)
	return states

def gen_one_traj_img(config):
	states = get_init_state(config)

	state_obs = []
	img_obs = []
	state_gt = []
	dones = []
	fails = []
	acs = []
	u_max = final_config.turnRate
	dt = config.dt
	v = config.speed

	xs = torch.tensor(config.obs_x)
	ys = torch.tensor(config.obs_y)
	rs = torch.tensor(config.obs_r)

	for t in range(config.data_length):
		# random between -u_max and u_max
		ac = torch.rand(1) * 2 * u_max - u_max

		states_next = torch.rand(3)
		states_next[0] = states[0] + v*dt*torch.cos(states[2])
		states_next[1] = states[1] + v*dt*torch.sin(states[2])
		states_next[2] = states[2] + dt*ac

		# the data is (o_t, a_t), don't observe o_t+1 yet
		state_obs.append(states[2].numpy()) # get to observe theta
		state_gt.append(states.numpy()) # gt state for debugging
		if t == config.data_length-1:
			dones.append(1)
		elif torch.abs(states[0]) > config.x_max-config.buffer or torch.abs(states[1]) > config.y_max-config.buffer: # out of bounds
			dones.append(1)
		else:
			dones.append(0)

		fails.append(torch.any(failure_check_batch(states, xs, ys, rs)).item())
		acs.append(ac)
		img_array = state_to_image_pil_hq(states, config)
		img_obs.append(img_array)
		states = states_next
		if dones[-1] == 1:
			break
	return state_obs, acs, state_gt, img_obs, dones, fails

def generate_trajs(config):
	demos = []
	for i in range(config.num_trajs):
		state_obs, acs, state_gt, img_obs, dones, fails = gen_one_traj_img(config)
		demo = {}
		demo['obs'] = {'image': img_obs, 'state': state_obs, 'priv_state': state_gt}
		demo['actions'] = acs
		demo['dones'] = dones
		demo['fails'] = fails
		demos.append(demo)
		print('demo: ', i, "timesteps: ", len(state_obs), end='\r')
	
	with open('wm_demos'+str(config.size[0])+'_gap.pkl', 'wb') as f:
		pickle.dump(demos, f)

def recursive_update(base, update):
	for key, value in update.items():
		if isinstance(value, dict) and key in base:
			recursive_update(base[key], value)
		else:
			base[key] = value

if __name__=='__main__':            
	parser = argparse.ArgumentParser()
	
	config, remaining = parser.parse_known_args()

	yaml = yaml.YAML(typ="safe", pure=True)
	configs = yaml.load(pathlib.Path(f"{cwd}/{config_path}").read_text())

	name_list = ["defaults"]

	defaults = {}
	for name in name_list:
		recursive_update(defaults, configs[name])
	parser = argparse.ArgumentParser()
	for key, value in sorted(defaults.items(), key=lambda x: x[0]):
		arg_type = tools.args_type(value)
		parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
	final_config = parser.parse_args(remaining)

	demos = generate_trajs(final_config)
