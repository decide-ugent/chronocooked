import torch as th
import torch.nn as nn
from typing import Tuple

from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from torch import Tensor


class CTRNNCell(nn.Module):
    """
    Continuous-Time RNN cell.
    τ * dh/dt = -h + tanh(W_hh @ h + W_ih @ x + b)
    Discretized with Euler: h(t+1) = h(t) + (dt/τ) * (-h(t) + tanh(...))
    """
    def __init__(self, input_size: int, hidden_size: int, dt: float = 0.1, tau: float = 1.0, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.dt = dt
        self.num_layers = num_layers
        # Learnable time constants
        self.tau = nn.Parameter(th.full((hidden_size,), fill_value=tau))


        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: th.Tensor, h: th.Tensor) -> tuple[Tensor, Tensor]:
        #  x: (seq_len, batch, input_size) OR (batch, input_size)
        #  h: (1, batch, hidden_size)
        # (sequence length, batch size, features dim)
        # (batch size = n_envs for data collection or n_seq when doing gradient update)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h = h.squeeze(0)
        tau = nn.functional.softplus(self.tau)  # (hidden_size,) always positive
        alpha = self.dt /  tau
        outputs = []

        for t in range(x.size(0)):
            pre_act = self.input_layer(x[t]) + self.hidden_layer(h)
            h = (1 - alpha) * h + alpha * th.tanh(pre_act)
            outputs.append(h.unsqueeze(0))

        out = th.cat(outputs, dim=0)  # (seq_len, batch, hidden_size)
        return out, h.unsqueeze(0)  # (1, batch, hidden_size)




class CTRNNPolicy(RecurrentActorCriticPolicy):
    """
      Replaces the LSTM in RecurrentActorCriticPolicy with a CTRNN cell.
      """

    def __init__(self, *args, **kwargs):
        self.lstm_hidden_size = kwargs.get("lstm_hidden_size", 125)

        super().__init__(*args, **kwargs)

        # Replace LSTM with CTRNN
        self.lstm_actor = CTRNNCell(
            input_size=self.features_dim,
            hidden_size=self.lstm_hidden_size
        )
        # assuming critic lstm is enabled (default)
        if self.enable_critic_lstm:
            self.lstm_critic = CTRNNCell(
                input_size=self.features_dim,
                hidden_size=self.lstm_hidden_size,
            )

    @staticmethod
    def _process_sequence(
            features: th.Tensor,
            lstm_states: tuple[th.Tensor, th.Tensor],  # (hidden, dummy_cell)
            episode_starts: th.Tensor,
            ctrnn: CTRNNCell
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:

        ctrnn_hidden_state, cell_state = lstm_states  # cell_state = dummy

        # (sequence length, batch size, features dim)
        # (batch size = n_envs for data collection or n_seq when doing gradient update)
        n_seq = ctrnn_hidden_state.shape[1]

        # reshape to sequence
        features_sequence = features.reshape((n_seq, -1, ctrnn.input_size)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        # fast path (no resets)
        if th.all(episode_starts == 0.0):
            ctrnn_output, ctrnn_hidden_state = ctrnn(features_sequence, ctrnn_hidden_state)

            ctrnn_output = th.flatten(ctrnn_output.transpose(0, 1), start_dim=0, end_dim=1)

            # return dummy cell unchanged
            return ctrnn_output, (ctrnn_hidden_state, cell_state)

        # slow path (with resets)
        outputs = []

        for feat, episode_start in zip(features_sequence, episode_starts, strict=True):
            reset_mask = (1.0 - episode_start).view(1, n_seq, 1)

            ctrnn_hidden_state = reset_mask * ctrnn_hidden_state  # reset only hidden

            out, ctrnn_hidden_state = ctrnn(
                feat.unsqueeze(0),
                ctrnn_hidden_state,
            )

            outputs.append(out)

        ctrnn_output = th.flatten(
            th.cat(outputs).transpose(0, 1),
            start_dim=0,
            end_dim=1,
        )

        return ctrnn_output, (ctrnn_hidden_state, cell_state)