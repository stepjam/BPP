from functools import partial
from typing import Tuple, Dict

import gym
import numpy as np
import torch
from stable_baselines3.common.distributions import \
    Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import Actor as SACActor, MlpPolicy
from torch import nn, Tensor

from models import PointNet
from bingham import torch_bingham, utils


class BinghamDistribution(Distribution):

    def __init__(self, device: torch.device):
        super(BinghamDistribution, self).__init__()
        self.distribution = torch_bingham.BinghamDistribution(device)

    def rescale_bingham(self, x):
        return torch.cat(
            [torch.clip(x[:, :3], -2, np.log(500)), torch.tanh(x[:, 3:])], 1)

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0
                               ) -> Tuple[nn.Module, nn.Parameter]:
        raise NotImplementedError('Not needed.')

    def proba_distribution(self, M: Tensor, Z: Tensor) -> "BinghamDistribution":
        self._M = M
        self._Z = Z
        return self

    def log_prob(self, actions: Tensor) -> Tensor:
        log_prob = self.distribution.log_probs(actions, self._M, self._Z)
        return log_prob

    def entropy(self) -> Tensor:
        e = self.distribution.entropy(self._M, self._Z)
        return e

    def sample(self) -> Tensor:
        s = self.distribution.rsample(self._M, self._Z)
        return s

    def mode(self) -> Tensor:
        return self._M[:, :, -1]

    def actions_from_params(self, M: Tensor, Z: Tensor,
                            deterministic: bool = False) -> Tensor:
        self.proba_distribution(M, Z)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, M: Tensor, Z: Tensor
                             ) -> Tuple[Tensor, Tensor]:
        actions = self.actions_from_params(M, Z)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class CustomSACActor(SACActor):

    def __init__(self, *args, **kwargs):
        super(CustomSACActor, self).__init__(*args, **kwargs)
        assert not self.use_sde
        self.log_std = self.mu = None
        del self.log_std
        del self.mu
        last_layer_dim = self.net_arch[-1] if len(
            self.net_arch) > 0 else self.features_dim
        self.vec19 = nn.Linear(last_layer_dim, 19)
        self.action_dist = None

    def get_std(self) -> Tensor:
        raise NotImplementedError('Not needed.')

    def reset_noise(self, batch_size: int = 1) -> None:
        raise NotImplementedError('Not needed.')

    def get_action_dist_params(self, obs: Tensor) -> Tuple[
        Tensor, Tensor, Dict[str, Tensor]]:
        if self.action_dist is None:
            self.action_dist = BinghamDistribution(self.device)
        features = self.extract_features(obs)
        latent_pi = self.latent_pi(features)
        vec19 = self.vec19(latent_pi)
        M, Z = utils.vec19_to_m_z(self.action_dist.rescale_bingham(vec19))
        return M, Z, {}

    def forward(self, obs: Tensor, deterministic: bool = False) -> Tensor:
        M, Z, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.actions_from_params(M, Z,
                                                    deterministic=deterministic,
                                                    **kwargs)

    def action_log_prob(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        M, Z, kwargs = self.get_action_dist_params(obs)
        a, lp = self.action_dist.log_prob_from_params(M, Z, **kwargs)
        return a, lp

    def _predict(self, observation: Tensor,
                 deterministic: bool = False) -> Tensor:
        return self.forward(observation, deterministic)


class CustomSACPolicy(MlpPolicy):
    def make_actor(self, features_extractor=None):
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor)
        return CustomSACActor(**actor_kwargs).to(self.device)


class CustomActorCriticPolicy(ActorCriticPolicy):

    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs)
        assert not self.use_sde

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_dist = None
        self.action_net = nn.Linear(latent_dim_pi, 19)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(),
                                              lr=lr_schedule(1),
                                              **self.optimizer_kwargs)

    def forward(self, obs: Tensor, deterministic: bool = False) -> Tuple[
        Tensor, Tensor, Tensor]:
        if self.action_dist is None:
            self.action_dist = BinghamDistribution(self.device)
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi: Tensor) -> Distribution:
        vec19 = self.action_net(latent_pi)
        M, Z = utils.vec19_to_m_z(
            self.action_dist.rescale_bingham(vec19))
        return self.action_dist.proba_distribution(M, Z)


class CustomCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        self._model = PointNet(features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self._model(observations)
