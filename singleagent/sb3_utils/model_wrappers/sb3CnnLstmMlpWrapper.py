import torch.nn as nn
import torch


# access intermediate layer outputs - cnnlstmmlp
class sb3CnnLstmMlpWrapper(nn.Module):
    def __init__(self, model):
        super(sb3CnnLstmMlpWrapper ,self).__init__()
        self.pi_cnn = model.policy.pi_features_extractor.cnn
        self.vf_cnn = model.policy.vf_features_extractor.cnn
        self.pi_features_extractor = model.policy.pi_features_extractor
        self.vf_features_extractor = model.policy.vf_features_extractor
        self.mlp_extractor = model.policy.mlp_extractor
        self.mlp_action_net = self.mlp_extractor.policy_net
        self.mlp_value_net = self.mlp_extractor.value_net
        self.action_net = model.policy.action_net
        self.value_net = model.policy.value_net
        self.lstm_actor = model.policy.lstm_actor
        self.lstm_hidden_size = model.policy.lstm_actor.hidden_size
        if model.policy.lstm_critic:
            self.lstm_critic = model.policy.lstm_critic
        else:
            self.lstm_critic = None

    def forward(self,x, critic_states=None ,actor_states=None, deterministic=True ):

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        # cnn layer outputs
        x_pi = x.detach().clone().float().permute(0, 3, 1, 2)
        x_vf = x.detach().clone().float().permute(0, 3, 1, 2)
        pi_cnn_layer_outputs = {}
        vf_cnn_layer_outputs = {}

        for i, layer in enumerate(self.pi_cnn):
            x_pi = layer(x_pi)
            pi_cnn_layer_outputs[i] = x_pi.clone().detach().numpy()

        for i, layer in enumerate(self.vf_cnn):
            x_vf = layer(x_vf)
            vf_cnn_layer_outputs[i] = x_vf.clone().detach().numpy()

        # output of all layers
        cnn_policy_features = self.pi_features_extractor(x)
        cnn_value_features = self.vf_features_extractor(x)

        if actor_states is None:
            actor_states = (torch.zeros( 1, self.lstm_hidden_size),torch.zeros( 1, self.lstm_hidden_size))
            critic_states = (torch.zeros( 1, self.lstm_hidden_size),torch.zeros( 1, self.lstm_hidden_size))
        # x = self.mlp_extractor(x)
        lstm_action_output, actor_states = self.lstm_actor(cnn_policy_features, actor_states) # this is different from model.predict. so mlp will give different behaviour than actual
        lstm_action_output = torch.flatten(lstm_action_output.transpose(0, 1), start_dim=0, end_dim=1)

        if self.lstm_critic:
            lstm_critic_output, critic_states = self.lstm_critic(cnn_value_features, critic_states)
        else:
            lstm_critic_output = lstm_action_output.detach()
            critic_states = (actor_states[0].detach(), actor_states[1].detach())

        x_value_mlp = self.mlp_value_net(lstm_critic_output)
        x_action_mlp = self.mlp_action_net(lstm_action_output)  # set neuron outputs to zero for selected neurons
        x_action = self.action_net(x_action_mlp)
        x_value = self.value_net(x_value_mlp)

        # distribution = model.policy._get_action_dist_from_latent(x_action_mlp)
        # actions = distribution.get_actions(deterministic=deterministic)
        # log_prob = distribution.log_prob(actions)

        return pi_cnn_layer_outputs, cnn_policy_features, vf_cnn_layer_outputs, cnn_value_features, lstm_action_output, actor_states, lstm_critic_output, critic_states, x_action_mlp, x_value_mlp, x_action, x_value
