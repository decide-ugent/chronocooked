
import torch.nn as nn
import torch as th
from sb3_contrib.ppo_recurrent.policies import RecurrentActorCriticPolicy

# Basic implementation to test ppo with gru instead of lstm

class CustomGRUPolicy(RecurrentActorCriticPolicy):
    """
    Replaces the LSTM in RecurrentActorCriticPolicy with a GRU.
    """

    def __init__(self, *args, **kwargs):
        self.lstm_hidden_size = kwargs.get("lstm_hidden_size", 125)

        super().__init__(*args, **kwargs)

        # Replace LSTM with GRU
        self.lstm_actor  = nn.GRU(
            input_size=self.features_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=1,
        )
        # assuming critic lstm is enabled (default)
        if self.enable_critic_lstm:
            self.lstm_critic  = nn.GRU(
                input_size=self.features_dim,
                hidden_size=self.lstm_hidden_size,
                num_layers=1,
            )

    @staticmethod
    def _process_sequence(
        features: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],  # (hidden, dummy_cell)
        episode_starts: th.Tensor,
        gru: nn.GRU,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:

        hidden_state, cell_state = lstm_states  # cell_state = dummy

        n_seq = hidden_state.shape[1]

        # reshape to sequence
        features_sequence = features.reshape((n_seq, -1, gru.input_size)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        # fast path (no resets)
        if th.all(episode_starts == 0.0):
            gru_output, hidden_state = gru(features_sequence, hidden_state)

            gru_output = th.flatten(gru_output.transpose(0, 1), start_dim=0, end_dim=1)

            # return dummy cell unchanged
            return gru_output, (hidden_state, cell_state)

        # slow path (with resets)
        outputs = []

        for feat, episode_start in zip(features_sequence, episode_starts, strict=True):
            reset_mask = (1.0 - episode_start).view(1, n_seq, 1)

            hidden_state = reset_mask * hidden_state  # reset only hidden

            out, hidden_state = gru(
                feat.unsqueeze(0),
                hidden_state,
            )

            outputs.append(out)

        gru_output = th.flatten(
            th.cat(outputs).transpose(0, 1),
            start_dim=0,
            end_dim=1,
        )

        return gru_output, (hidden_state, cell_state)