def main(continue_training=True):
#def main():

  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['xlarge'])#First Change Model Size
  config = config.update({
      'logdir': '~/logdir/TR64/large', #Second Change Logdir
      'run.train_ratio': 16, #Thirt Change
      'run.log_every': 60,  # Seconds
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
  # Define ObstacleTower configs
  config1 = {
    "train-mode": 1,             # Training mode
    "tower-seed": -1,            # Random tower on every reset
    "starting-floor": 0,        # Starting floor
    "total-floors": 100,        # Maximum number of possible floors
    "dense-reward": 1,           # Use sparse reward
    "lighting-type": 1,          # Single realtime light with minimal color variations
    "visual-theme": 1,           # Normal ordering of themes
    "agent-perspective": 0,      # First-person perspective
    "allowed-rooms": 2,         # Use normal, key, and puzzle rooms
    "allowed-modules": 2,       # Use the full range of modules
    "allowed-floors": 2,        # Include layouts with branching and circling
    "default-theme": 0,          # Default theme set to Ancient
    }

  print('\nconfig',config) #I add this line
  from embodied.envs import from_gym
  from ObstacleTowerEnv import ObstacleTowerEnv #I add this line
  env = ObstacleTowerEnv(retro=True, realtime_mode=True)  # Replace this with your Gym env.
  obs = env.reset(config=config1)#Obstacle Tower Configs
  env = from_gym.FromGym(env, obs_key='image')  # Or obs_key='vector'.
  env = dreamerv3.wrap_env(env, config)
  env = embodied.BatchEnv([env], parallel=False)

  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  replay = embodied.replay.Uniform(
      config.batch_length, config.replay_size, logdir / 'replay')
  #if continue_training:
        # Φορτώστε την αποθηκευμένη μνήμη αναπαραγωγής αν θέλετε να συνεχίσετε
        #replay.load() 
  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)
  embodied.run.train(agent, env, replay, logger, args)
  #embodied.run.eval_only(agent, env, logger, args)


if __name__ == '__main__':
  continue_training = True
  main(continue_training)
  #main()
