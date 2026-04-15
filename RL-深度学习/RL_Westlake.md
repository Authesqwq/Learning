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
	==当前状态价值 = 当前一步期望奖励 + 折现后的下一状态价值。==
## Chapter 3. Optimal Policy and Bellman Optimality Equation(BOE)

- If:$$
  v_{\pi_1}(s) \ge v_{\pi_2}(s), \forall s \in S
  $$
  then $\pi_1$ is better than$\pi_2$.

  A policy $\pi^*$ is optimal if $v_{\pi^*}(s) \ge v_{\pi_2}(s)$.$$
  \begin{aligned}
  v(s)&=\max_{\pi}\sum_{a} \pi(a\mid s)[\sum_{r} p(r\mid s,a)r+\gamma \sum_{s'}p(s'\mid s,a)v_{\pi}(s')], \forall s \in S\\
  &=\max_{\pi}\sum_{a} \pi(a\mid s)q(s,a) \;\; s\in S
  \end{aligned}
  $$
 - Bellman optimality equation:$$
  v=\max_{\pi}(r_{\pi}+\gamma P_{\pi}v)
  $$
	Let$$
  f(v)=\max_{\pi}(r_{\pi}+\gamma P_{\pi}v)
  $$Then it becomes $v=f(v)$
	  where$$
	  [f(v)]_s=\sum_{a}\pi(a\mid s)q_{\pi}(s,a),\;\; s\in S$$
  - Contraction mapping theorem
	  - Fixed point: $x\in X$ is a fixed point of $f: X \to X$ if $f(x)=x$
	  - contraction mapping(or contractive function):$f$ is a cm if$$||f(x_1)-f(x_2)||\le \gamma||x_1 - x_2||$$where $\gamma \in (0,1)$
	  - Theorem:
		  - For any equation that has the form of $x=f(x)$,then
		  - Existence: there exists a fixed point $x^*$ satisfying $f(x^*)=x^*$
		  - Uniqueness: the fixed point $x^*$ is unique
		  - Algorithm: consider a sequence $\{x_k\}$ where$x_{k+1}=f(x_k)$,then $x_k \to x^*$ as $k\to \infty$
		  - ==压缩映射？==
	  - Solve the BOE
  - Policy optimality
	  - Suppose$$\pi^* = arg \max{\pi}(r_{\pi}+\gamma P_{\pi}v^*)$$Then$$v^*=r_{\pi^*}+\gamma P_{\pi^*}v^*$$Therefore, $\pi^*$ is a policy and $v^* = v_{\pi^*}$ is the corresponding state value.
	  - Greedy Optimal Policy
  - Factors:
	  - ![[Pasted image 20260414155347.png]]
	  - we know the red factors:
		  - Reward design:$r$
		  - System model:$p(s'|s,a)\; , \; p(r|s,a)$
		  - Discount rate:$\gamma$, the bigger the more long-sighted
	  - Optimal Policy Invariance: $r\to ar+b$would not change $v^*$
	  - meaningless detour: if we set r=0, $\gamma$ will still limit detouring

## Chapter 4: Value Iteration & Policy Iteration

- Value iteration algorithm
	- The algorithm$$v_{k+1}=f(v_k)=\max{\pi}(r_{\pi}+\gamma P_{\pi}v_k),\; k=1,2...$$start from $v_0$
	- Step1: policy update. This step is to solve$$\pi_{k+1}=arg\max{\pi}(r_{\pi}+\gamma P_{\pi}v_k)$$where $v_k$ is given.
	- Step2: value update.$$v_{k+1}={r_{\pi_{k+1} }{+\gamma P{\pi_{k+1}}}v_k}$$
	- Procedure: $$v_{k}(s)\to q_{k}(s,a)\to greedy\; policy \;\pi_{k+1}(a|s)\to new \; value \; v_{k+1}=\max_{a} q_{k}(s,a)$$

- Policy iteration algorithm
	- start from $\pi_0$
	- step1: policy evaluation(PE)$$v_{\pi_k}=r_{\pi_k}+\gamma P_{\pi_k}v_{\pi_k}$$
	- step2: policy improvement(PI)$$\pi_{k+1}=arg\max_{\pi}(r_{\pi}+\gamma P_{\pi}v_{\pi_k})$$
	- Only in theory

- Truncated policy iteration algorithm
	- ==不判断是否收敛，而判断是否执行足够多次数==

## Chapter 5: Monte Carlo Learning

- Without models: Monte Carlo estimation
	- imposible to know the distribution
	- MC estimation: $\mathbb{E}[x]\approx \bar{x}=\frac{1}{N}\sum_{j=1}^{N} x_j$
	- Law of Large Numbers
	- $$q_{\pi_k}(s,a)=\mathbb{E}[G_t|S_t=s,A_t=a]\approx\frac{1}{N}\sum_{i=1}^{N}g^(i)(s,a)$$
	- Fundamental idea: When model is unavailable, we can use data(sample/experience)
- MC Basic algorithm
	- Step1: PE. Estimate $q_{\pi_k}(s,a)$ directly, instead of solving $v_{\pi_k}(s)$.
	- Step2: PI. No change.
	- Low efficiency
	- episode length: should be sufficiently long
- MC Exploring Starts
	- **Visit**: every time a state-action pair appears in the episode, it is called a visit of that s-a pair
	- data-efficient method: first-visit method; every-visit method
	- Generalized policy iteration: switching between PE and PI
	- Still difficult to achieve
- MC $\epsilon$-greedy
	- $\epsilon$-greedy policy$$\pi(a|s)=1-\frac{\epsilon}{|A(s)|}(|A(s)|-1)$$, for the greedy action.
		- balance between exploitation and exploration: when $\epsilon=0$, it becomes greedy; when $\epsilon=1$, it becomes a uniform distribution.
	- select $\pi\in {\Pi}_{\epsilon}$
	- Advantage: stronger exploration ablity
	- Disadvantage: not optimal in gereral
## Chapter6. Stochastic Approximation and Stochastic Gradient Descent

- Mean estimation
	- how to calculate the $\bar{x}$?
		- collect all
		- incremental and iterative manner
			- suppose $w_{k+1}=\frac{1}{k}\sum_{i=1}^{k}x_i$
			then,$$w_{k+1}=w_k-\frac{1}{k}(w_k-x_k)$$it is better than nothing.
			- let $\alpha_k$ replace 1/k.
- Stochastic approximation: SA
	- a broad class of stochastic iterative algorithms
	- it does not require to know the expression of the objective function
- Robbins-Monro algorithm: RM
	- Problem statement:$$g(w)=\nabla_{w}J(w)=0$$if we do not know the expression of $g$?==神经网络==$$w_{k+1}=w_k-a_k\tilde{g}(w_k,\eta_{k})$$
		- $\tilde{g}(w_k,\eta_{k})=g(w_k)+\eta_k$ is the $k$th noisy observation
		- $a_k$ is a positive coefficient
		- thus, the function $g(w)$ is a black box!
		- PHILOSOPHY: without model, we need data!
	- Convergence properties:
		- three conditions:
			- $0\lt c_1\le\nabla_w g(w)\le c_2$
				- g to be monotonically increasing; the gradient is bounded from the above.
			-  $\sum_{k=1}^{\infty}a_k=\infty$and$\sum_{k=1}^{\infty}a_k^2\le \infty$ ensures that $a_k$ converges to 0 as $k\to\infty$; and $a_k$ do not converge to 0 too fast.
			- $\mathbb{E}[\eta_{k}|H_k]=0$ and $\mathbb{E}[\eta_{k}^2|H_k]\lt \infty$ .
		- a typical sequence is $a_k=\frac{1}{k}$. we will see that $a_k$ is often selected as a sufficiently small constant.
	- Apply to mean estimation
		- Dvoretzky's theorem