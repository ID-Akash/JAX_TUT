import functools
from typing import Dict, Tuple
import jax
import jumanji

# Instantiate a Jumanji environment using the registry
from a2c_agent import A2CAgent
from actor_critic_network import make_actor_critic_networks_snake
from optax import adam
from jumanji.training.types import TrainingState, ActingState
from jumanji.wrappers import VmapAutoResetWrapper
import jax.numpy as jnp
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    # Make raw environment
    env = jumanji.make("Snake-v1")
    # Reset your (jit-able) environment
    key = jax.random.PRNGKey(0)

    # A wrapper to instantiate multiple environments in parallel
    env = VmapAutoResetWrapper(env)

    # Setup Agent
    actor_critic_networks = make_actor_critic_networks_snake(
        snake=env,
        num_channels=cfg.network.num_channels,
        policy_layers=cfg.network.policy_layers,
        value_layers=cfg.network.value_layers,
    )

    optimiser = adam(learning_rate=cfg.a2c.learning_rate)
    # TODO: Figure out what's normalize_advantage and bootstrapping_factor. Also, whats the role of batch_size?
    rl_agent = A2CAgent(
        env=env,
        n_steps=cfg.training.n_steps,
        total_batch_size=cfg.training.total_batch_size,
        optimizer=optimiser,
        actor_critic_networks=actor_critic_networks,
        normalize_advantage=True,
        discount_factor=cfg.a2c.discount_factor,
        bootstrapping_factor=cfg.a2c.bootstrapping_factor,
        l_pg=cfg.a2c.l_pg,
        l_td=cfg.a2c.l_td,
        l_en=cfg.a2c.l_en,
    )

    params_key, reset_key, acting_key = jax.random.split(key, 3)

    # Initialize params.
    params_state = rl_agent.init_params(params_key)

    # Initialize environment states.
    num_local_devices = jax.local_device_count()
    num_global_devices = jax.device_count()

    # Assumption: All workers have same number of local devices
    num_workers = num_global_devices // num_local_devices

    # total_batch_size acheived my collectively aggregating data from all the devices
    local_batch_size = rl_agent.total_batch_size // num_global_devices

    # num_reset_keys = total number of possible resets
    # (num_workers, num_local_devices, local_batch_size, 2)
    reset_keys = jax.random.split(reset_key, rl_agent.total_batch_size).reshape(
        (
            num_workers,
            num_local_devices,
            local_batch_size,
            -1,
        )
    )

    # Each worker can only control it's local devices. Therefore,
    # we extract the resets keys for each worker. For each worker this program
    # will have to be run manually

    # (num_local_devices, local_batch_size, 2)
    reset_keys_per_worker = reset_keys[jax.process_index()]

    env_state, timestep = jax.pmap(env.reset, axis_name="devices")(
        reset_keys_per_worker
    )

    # Initialize acting states
    acting_key_per_device = jax.random.split(acting_key, num_global_devices).reshape(
        num_workers, num_local_devices, -1
    )
    # (num_local_devices, local_batch_size, 2)
    acting_key_per_worker_device = acting_key_per_device[jax.process_index()]

    acting_state = ActingState(
        state=env_state,
        timestep=timestep,
        key=acting_key_per_worker_device,
        episode_count=jnp.zeros(num_local_devices, float),
        env_step_count=jnp.zeros(num_local_devices, float),
    )

    # Build the training state.
    training_state = TrainingState(
        params_state=jax.device_put_replicated(params_state, jax.local_devices()),
        acting_state=acting_state,
    )

    # This function is used to pmap the run_epoch functions across multiple local devices
    @functools.partial(jax.pmap, axis_name="devices")
    def epoch_fn(training_state: TrainingState) -> Tuple[TrainingState, Dict]:
        training_state, metrics = jax.lax.scan(
            f=lambda training_state, _: rl_agent.run_epoch(training_state),
            init=training_state,
            xs=None,
            length=cfg.training.num_learner_steps_per_epoch,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)

        return training_state, metrics

    # num_epochs = 10
    for i in range(cfg.training.num_epochs):
        training_state, metrics = epoch_fn(training_state)
        # env.render(training_state.acting_state.state)

    print(metrics)


if __name__ == "__main__":
    train()
