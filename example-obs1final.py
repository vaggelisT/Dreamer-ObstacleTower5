import gym
from gym.spaces import Box
import numpy as np
from PIL import Image
from gym import spaces
import cv2

class ActionSpaceWrapper(gym.ActionWrapper):
    def __init__(self, env, allowed_actions):
        super().__init__(env)
        self.allowed_actions = allowed_actions

        # Define a new discrete action space with the reduced number of actions
        self.action_space = gym.spaces.Discrete(len(allowed_actions))

    def action(self, action):
        # Map the action to the original action space based on the allowed actions
        original_action = self.allowed_actions[action]
        return original_action

class ResetWrapper(gym.Wrapper):
    def __init__(self, env, allowed_actions):
        super().__init__(env)
        self.allowed_actions = allowed_actions

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info  # Return only the observation


#def main(continue_training=True):
def main():


  from matplotlib import pyplot as plt
  import warnings
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['medium'])
  config = config.update({
      'logdir': '~/logdir/run1',
      'run.train_ratio': 64,
      'run.log_every': 30,  # Seconds
      'batch_size': 16,
      'jax.prealloc': False,
      'encoder.mlp_keys': '$^',
      'decoder.mlp_keys': '$^',
      'encoder.cnn_keys': 'image',
      'decoder.cnn_keys': 'image',
      # 'jax.platform': 'cpu',
  })
  config = embodied.Flags(config).parse()

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
      # embodied.logger.WandBOutput(logdir.name, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ])
  
  # Define your configuration dictionary with desired parameter values
  config1 = {
    "train-mode": 0,             # Training mode
    "tower-seed": -1,            # Random tower on every reset
    "starting-floor": 0,        # Starting floor
    "total-floors": 100,        # Maximum number of possible floors
    "dense-reward": 1,           # Use sparse reward
    "lighting-type": 1,          # Single realtime light with minimal color variations
    "visual-theme": 1,           # Normal ordering of themes
    "agent-perspective": 1,      # First-person perspective
    "allowed-rooms": 2,         # Use normal, key, and puzzle rooms
    "allowed-modules": 2,       # Use the full range of modules
    "allowed-floors": 0,        # Include layouts with branching and circling
     #"visual-theme": 2,
    "default-theme": 2,          # Default theme set to Ancient
     #"use-industrial":1
    }

  from embodied.envs import from_gym
  from ObstacleTowerEnv import ObstacleTowerEnv
  # breakpoint()
  env = ObstacleTowerEnv(retro=True, realtime_mode=True)
  obs = env.reset(config=config1)
  #====================Action Space changes for experiments========================
  #allowed_actions = list(range(17))  # Assuming you want to allow action indices from 0 to 16
  #allowed_actions = [3, 6, 9, 12, 15, 18, 21,24,27,30,33, 36, 39, 42, 45,48,51]  # Example: Allow only even-numbered actions from 0 to 10
  #env = ActionSpaceWrapper(env, allowed_actions)
   #================================================================
  env = from_gym.FromGym(env, obs_key='image')  # Or obs_key='vector'.
  print('\nconfig',config)
  #print('---------------Meta to FromGym------------------------')
  #print("Meta to Fromgym to Env einai=",env)
  #print(type(env))
  #print("Meta to Fromgym to Env einai=",env)

  env = dreamerv3.wrap_env(env, config)
  env = embodied.BatchEnv([env], parallel=False)

  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  replay = embodied.replay.Uniform(
      config.batch_length, config.replay_size, logdir / 'replay')
  #=================================En theloume na sinexisoume to Training apo eki pou miname
  #if continue_training:
        # Φορτώστε την αποθηκευμένη μνήμη αναπαραγωγής αν θέλετε να συνεχίσετε
        #replay.load() 
  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)
  embodied.run.train(agent, env, replay, logger, args)
  # embodied.run.eval_only(agent, env, logger, args)


if __name__ == '__main__':
  #continue_training = True
  #main(continue_training)
  main()
