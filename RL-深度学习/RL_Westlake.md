# 强化学习的数学原理 BY赵世钰

目录：1.基本概念；2.贝尔曼公式；3.贝尔曼最优公式；4.值策略迭代；5.蒙特卡罗方法；6.随机近似理论；7.时序差分方法；8.值函数近似；9.策略梯度方法；10.Actor-Critic方法



## Chapter1. Basic Concepts

- **State**: the status of the agent with respect to the environment.

- **State space**: the set of all states $S$

- **Action:** for each state, ...

  Action space of a state: $A(s_i)={a_i}^?$

- **State transition:** $s_1  \xrightarrow{a_1}  s_2$

  tabular representation: only deterministic

- **State transition probability**: at $s_1$, if we choose $a_2$, the next state is $s_2$

$$
p(s_2|s_1,a_2)=1\\
p(s_i|s_1,a_2)=0 \quad \forall i \ne 2
$$

​      could be stochastic.

- **Policy**: what actions to take at a state. 

  for example, for state $s_1$:

$$
\pi(a_1|s_1)=0 \\
\pi(a_2|s_1)=1 \\
...
$$

- **Reward**: a real number we get after taking an action.

  positive--encouragement

  negative--punishment

  (can be reversed) as a **human-machine interface**.

  At $s_1$, if we choose $a_1$, the reward is -1:

$$
p(r=-1|s_1,a_1)=1 \;and \;p(r\ne-1|s_1,a_1)=0
$$

- **Trajectory**: a state-action-reward chain.

- **Return** of trajectory: sum of the reward.

  **Discounted return**: introduce a discount rate $\gamma \in [0,1)$

$$
discounted return=r_1+\gamma r_2+\gamma^2 r_3+...
$$

- **Episode**: or trail. The agent should stop at some terminal states. An episode is usually assumed to be a finite trajectory.

  Tasks with no terminal states are called continuing tasks, else called episodic tasks.

  treat the target state as a special absorbing state/ normal state.

****

#### **Markov decision process(MDP)**:

- **Sets:** State $S$, Action $A(s),\;s\in S$, Reward $R(s,a)$

- **Probability distribution:** 

  State transition prob: at $s$, taking $a$, the prob to transit to state $s'$ is $p(s'|s,a)$

  Reward prob: at $s$, taking $a$, the prob to get reward $r$ is $p(r|s,a)$

- **Policy**: at $s$, the prob to choose $a$ is $\pi(a|s)$

- **Markov property (memoryless property)**:

$$
p(s_{t+1}|a_{t+1},s_t,...,a_1,s_0)=p(s_{t+1}|a_{t+1},s_t)\\
p(r_{t+1}|a_{t+1},s_t,...,a_1,s_0)=p(r_{t+1}|a_{t+1},s_t)
$$

****

## Chapter 2. Bellman Equation

- how to calculate return?

  Method1: by definition. $v_1=r_1+\gamma r2+...$

  Method2: $v_1=r_1+\gamma v_2$ , $v_2=r_2+\gamma v_3$ ,... The returns rely on each other. **Bootstrapping!** 

  matrix-vector form:  $v=r+\gamma Pv$ (simple bellman equation)

- **Single-step process**： $S_t \xrightarrow{A_t} R_{t+1},S_{t+1}$ . They are all RVs.

  This step is governed by the following prob distributions:

  $S_t\to A_t$ is governed by $\pi (A_t=a|S_t=s)$

  $S_t,A_t\to R_{t+1}$ is governed by $p(R_{t+1}=r|S_t=s,A_t=a)$

  $S_t,A_t\to S_{t+1}$ is governed by $p(S_{t+1}=s'|S_t=s,A_t=a)$

- **Multi-step trajectory**: $S_t \xrightarrow{A_t} R_{t+1},S_{t+1}$, $S_{t+1} \xrightarrow{A_{t+1}} R_{t+2},S_{t+2}$,...

  The discounted return is  $G_t = R_{t+1}+\gamma R_{t+2}+...$

- **State value**: the expectation of $G_t$. 
  $$
  v_{\pi}(s)=\mathbb{E}[G_t|S_t=s]
  $$

  - It is a function of $s$.

  - It is based on the policy $\pi$.

  - Return is for single trajectory; State value is for multiple trajectories.

- **Bellman equation**: describes the relationship among the values of all states.

  Consider a random trajectory: $S_t \xrightarrow{A_t} R_{t+1},S_{t+1}$, $S_{t+1} \xrightarrow{A_{t+1}} R_{t+2},S_{t+2}$,...

  The return $G_t=R_{t+1}+\gamma G_{t+1}$

  Then, 
  $$
  \begin{aligned}
  v_{\pi}(s) &= \mathbb{E}[G_t \mid S_t = s] \\
  &= \mathbb{E}[R_{t+1} \mid S_t = s] + \gamma \mathbb{E}[G_{t+1} \mid S_t = s]
  \end{aligned}
  $$

  - First, calculate 
    $$
    \begin{aligned}
    \mathbb{E}[R_{t+1} \mid S_t = s]&= \sum_{a} \pi(a\mid s) \mathbb{E}[R_{t+1}\mid S_t=s,A_t=a]\\
    &=\sum_{a} \pi(a\mid s)\sum_{r} p(r\mid s,a)r
    \end{aligned}
    $$
    This is the mean of immediate rewards.

  - Second, calculate
    $$
    \begin{aligned}
    \mathbb{E}[G_{t+1}\mid S_t=s]&= \sum_{s'}\mathbb{E}[G_{t+1},G_{t+1}=s']p(s'\mid s)\\
    &=\sum_{s'}v_{\pi}(s')p(s'\mid s)\\
    &=\sum_{s'}v_{\pi}(s')\sum_{a}p(s'\mid s,a)\pi(a\mid s)
    \end{aligned}
    $$
    This is the mean of future rewards.

  - Therefore, we have
    $$
    v_{\pi}(s)=\sum_{a} \pi(a\mid s)[\sum_{r} p(r\mid s,a)r+\gamma \sum_{s'}p(s'\mid s,a)v_{\pi}(s')], \forall s \in S
    $$

    - $v_{\pi}(s)$ and $v_{\pi}(s')$ are state values to be calculated. Bootstrapping!
    - $\pi(a\mid s)$ is a given policy. Solving it is called policy evaluation.
    - $p(r\mid s,a)$ and $p(s'\mid s,a)$ represent the dynamic model. Maybe the model is known or not.

- **matrix-vector form**:

  rewrite it:
  $$
  v_{\pi}(s)=r_{\pi}(s)+\gamma \sum_{s'}p_{\pi}(s'\mid s)v_{\pi}(s')\\
  \rightarrow v_{\pi}=r_{\pi}+\gamma P_{\pi}v_{\pi}
  $$
  $P_{\pi}$ is state transition matrix.

- **Solve state values**: Given a policy, finding out the corresponding state values is called **policy evaluation**.

  - The closed-form solution:
    $$
    v_{\pi}=(I-\gamma P_{\pi})^{-1}r_{\pi}
    $$

  - An iterative solution:
    $$
    v_{k+1}=r_{\pi}+\gamma P_{\pi}v_{k}
    $$
    This algorithm leads to a sequence ${v_0,v_1,...}$. We can prove that
    $$
    V_k \to v_{\pi}=(I-\gamma P_{\pi})^{-1}r_{\pi},\; k\to \infty
    $$

- **Action value**: the average return the agent can get starting from a state and taking an action.

​	Definition:
$$
q_{\pi}(s,a)=\mathbb{E}[G_t\mid S_t=s,S_t=a]
$$
​	Hence,
$$
v_{\pi}(s)=\sum_{a}\pi(a\mid s)q_{\pi}(s,a)
$$

## Chapter 3. Optimal Policy and Bellman Optimality Equation(BOE)

- If:
  $$
  v_{\pi_1}(s) \ge v_{\pi_2}(s), \forall s \in S
  $$
  then $\pi_1$ is better than$\pi_2$.

  A policy $\pi^*$ is optimal if $v_{\pi^*}(s) \ge v_{\pi_2}(s)$.

  

- $$
  \begin{aligned}
  v(s)&=\max_{\pi}\sum_{a} \pi(a\mid s)[\sum_{r} p(r\mid s,a)r+\gamma \sum_{s'}p(s'\mid s,a)v_{\pi}(s')], \forall s \in S\\
  &=\max_{\pi}\sum_{a} \pi(a\mid s)q(s,a) \;\; s\in S
  \end{aligned}
  $$

  

- $$
  v=\max_{\pi}(r_{\pi}+\gamma P_{\pi}v)
  $$

  

- 