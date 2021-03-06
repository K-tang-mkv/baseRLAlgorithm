# Model-based method: dynamic programming

Dynamic programming assume that there is a perfect model given by the env.
There are two main algorithms of Dynamic programming, which are policy iteration and value iteration.

- ###policy iteration:
    This algorithm is an iterative algorithm that alternates between two main steps: policy evaluation and policy improvement.
  At each iteration, the current policy πk is evaluated estimating the action-value function Q(s, a) and the new policy
  π(k+1) is generated by taking greedy policy, i.e., select the best action with given state according to the Q-value.
  The process of the policy improvement is actually like, to monotonically increasing the policy that finally reach the 
  optimal policy based on the greedy selection according to the Q-value.
  The meaning of the policy evaluate is to say that use the policy improved to estimate the state values so that to evaluate 
  whether the policy is good after improving.
  
- ###value iteration:
    This algorithm means that we don't need to evaluate the policy after each policy improvement.
  That means we just do one time evaluation, and then generate the deterministic policy according to the greedy selection.
  This algorithm is of more convergence speed than the policy iteration method.