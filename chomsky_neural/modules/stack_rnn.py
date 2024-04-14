from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# First element is the stacks, second is the hidden internal state.
StackRnnState = Tuple[torch.Tensor, torch.Tensor]

# Number of actions the stack-RNN can take, namely POP, PUSH and NO_OP.
_NUM_ACTIONS = 3


def _update_stack(
    stack: torch.Tensor, actions: torch.Tensor, push_value: torch.Tensor
) -> torch.Tensor:
    """Updates the stack values."""
    batch_size, stack_size, stack_cell_size = stack.size()
    cell_tiled_stack_actions = actions.unsqueeze(1).repeat(
        1, stack_cell_size, 1
    )  # (batch_size, stack_cell_size, _NUM_ACTIONS)

    push_action = cell_tiled_stack_actions[..., 0]
    pop_action = cell_tiled_stack_actions[..., 1]
    pop_value = stack[..., 1, :]
    no_op_action = cell_tiled_stack_actions[..., 2]
    no_op_value = stack[..., 0, :]

    top_new_stack = (
        push_action * push_value + pop_action * pop_value + no_op_action * no_op_value
    )  # (batch_size, stack_cell_size)
    top_new_stack = top_new_stack.unsqueeze(1)  # (batch_size, 1, stack_cell_size)

    stack_tiled_stack_actions = (
        actions.unsqueeze(1).unsqueeze(2).repeat(1, stack_size - 1, stack_cell_size, 1)
    )  # (batch_size, stack_size - 1, stack_cell_size, _NUM_ACTIONS)
    push_action = stack_tiled_stack_actions[
        ..., 0
    ]  # (batch_size, stack_size - 1, stack_cell_size)
    push_value = stack[..., :-1, :]  # (batch_size, stack_size - 1, stack_cell_size)
    pop_action = stack_tiled_stack_actions[
        ..., 1
    ]  # (batch_size, stack_size - 1, stack_cell_size)
    pop_extra_zeros = torch.zeros(batch_size, 1, stack_cell_size).to(stack.device)
    pop_value = torch.cat(
        [stack[..., 2:, :], pop_extra_zeros], dim=1
    )  # (batch_size, stack_size - 1, stack_cell_size)
    no_op_action = stack_tiled_stack_actions[
        ..., 2
    ]  # (batch_size, stack_size - 1, stack_cell_size)
    no_op_value = stack[..., 1:, :]  # (batch_size, stack_size - 1, stack_cell_size)

    rest_new_stack = (
        push_action * push_value + pop_action * pop_value + no_op_action * no_op_value
    )

    return torch.cat([top_new_stack, rest_new_stack], dim=1)


class StackRNN(nn.Module):
    """Core for the stack RNN."""

    def __init__(
        self,
        inner_core: nn.Module,
        stack_cell_size: int,
        stack_size: int = 30,
        n_stacks: int = 1,
    ):
        """Initializes."""
        super().__init__()
        self.inner_core = inner_core
        self.stack_cell_size = stack_cell_size
        self.stack_size = stack_size
        self.n_stacks = n_stacks
        self.push_update_layer = nn.Linear(
            in_features=int(inner_core.hidden_size),
            out_features=self.n_stacks * self.stack_cell_size,
        )
        self.action_layer = nn.Linear(
            in_features=int(inner_core.hidden_size),
            out_features=self.n_stacks * _NUM_ACTIONS,
        )

    def forward(
        self, inputs: torch.Tensor, prev_state: StackRnnState
    ) -> Tuple[torch.Tensor, StackRnnState]:
        """Steps the stack RNN core.

        See base class docstring.

        Args:
        inputs: An input array of shape (batch_size, input_size). The time
            dimension is not included since it is an RNNCore, which is unrolled over
            the time dimension.
        prev_state: A _StackRnnState tuple, consisting of the previous stacks and
            the previous state of the inner core. Each stack has shape (batch_size,
            stack_size, stack_cell_size), such that `stack[n][0]` represents the top
            of the stack for the nth batch item, and `stack[n][-1]` the bottom of
            the stack. The stacks are just the concatenation of all these tensors.

        Returns:
        - output: An output array of shape (batch_size, output_size).
        - next_state: Same format as prev_state.
        """
        stacks, old_core_state = prev_state

        batch_size = stacks.size(0)
        top_stacks = stacks[:, :, 0, :].reshape(
            batch_size, self.n_stacks * self.stack_cell_size
        )

        inputs = torch.cat([inputs, top_stacks], dim=-1).unsqueeze(
            1
        )  # (batch_size, 1, input_size + n_stacks * stack_cell_size)
        new_core_output, new_core_state = self.inner_core(inputs, old_core_state)
        push_values = self.push_update_layer(new_core_output)
        push_values = push_values.reshape(
            batch_size, self.n_stacks, self.stack_cell_size
        )
        stack_actions = F.softmax(self.action_layer(new_core_output), dim=-1)
        stack_actions = stack_actions.reshape(batch_size, self.n_stacks, _NUM_ACTIONS)
        new_stacks_list = []
        for i in range(stacks.size(1)):
            new_stacks_list.append(
                _update_stack(
                    stacks[:, i, :, :], stack_actions[:, i, :], push_values[:, i, :]
                )
            )
        new_stacks = torch.stack(new_stacks_list, dim=1)

        return new_core_output, (new_stacks, new_core_state)
