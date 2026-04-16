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
	- impossible to know the distribution
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
	- Advantage: stronger exploration ability
	- Disadvantage: not optimal in general
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
		- $\tilde{g}(w_k,\eta_{k})=g(w_k)+\eta_k$ is the $k$-th noisy observation
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
- Stochastic Gradient Descent algorithm: SGD
	- solve: $$\min{w} J(w)=\mathbb{E}[f(w,X)]$$
	- Method 1: gradient descent GD:
		- $w_{k+1}=w_k-{\alpha}_k\nabla_w\mathbb{E}[f(w_k,X)]$
	- Method 2: batch gradient descent BGD:
		- $\mathbb{E}[\nabla_wf(w_k,X)]\approx\frac{1}{n}\sum_{i=1}^{n}\nabla_w f(w_k,x_i)$
	- Method 3: stochastic gradient descent SGD:
		- $w_{k+1}=w_k-{\alpha}_k\nabla_w f(w_k,x_k)$
		- replace the true gradient by the stochastic gradient
		- and let $n=1$
	- noisy measurement:  $\nabla_w f(w_k,x_k)=\mathbb{E}[\nabla_w f(w,X)]+\eta$
	- convergence: SGD is a special RM algorithm
	- relative error: $$\begin{aligned}\delta_k &= \frac{|\nabla_w f(w_k,x_k)-\mathbb{E}[\nabla_w f(w_k,X)]|}{|E[\nabla_w f(w_k,X)]|}\\&\le\frac{|\nabla_w f(w_k,x_k)-\mathbb{E}[\nabla_w f(w_k,X)]|}{c|w_k-w^*|}\end{aligned}$$
	- convert the deterministic formulation to the stochastic formulation of SGD. 
- BGD, MBGD, SGD:
	- BGD: $w_{k+1}=w_k-{\alpha}_k \frac{1}{n}\sum_{i=1}^{n}\nabla_w[f(w_k,x_i)]$
	- MBGD: $w_{k+1}=w_k-{\alpha}_k \frac{1}{m}\sum_{j\in I_k}\nabla_w[f(w_k,x_j)]$
	- SGD: $w_{k+1}=w_k-{\alpha}_k \nabla_w[f(w_k,x_k)]$
	- MBGD will be more flexible and efficient.

## Chapter7. Temporal-Difference Learning
- TD learning of state values: require the data/experience
	- ${(s_t,r_{t+1},s_{t+1})}_t$ ,generated following the given policy $\pi$.
	- (1) $$v_{t+1}(s_t)=v_t(s_t)-\alpha_t(s_t)\left[v_t(s_t)-[r_{t+1}+\gamma v_t(s_{t+1})]\right]$$(2) $v_{t+1}(s)=v_t(s), \forall s \ne s_t$
	- TD target $\bar{v}_t$: the algorithm drives $v(s_t)$ towards $\bar{v}_t$.
	- TD error $\delta_t=v_t(s_t)-[r_{t+1}+\gamma v_t(s_{t+1})]$: it is a difference between 2 consequent time steps. It reflects the deficiency between $v_t$ and $v_{\pi}$.
	- TD algorithm only estimates the state value of a given policy. It solves Bellman equation without the model(by experience).
	- TD learning is online, can handle both episodic and continuing tasks, is bootstrapping, has lower estimation variance. MC learning is offline, can only handle episodic tasks, is non-bootstrapping, has higher estimation variance.
- Sarsa: TD learning of action values
	- suppose$(s_t,a_t,r_{t+1},s_{t+1},a_{t+1})$, ${(s,a)}\to q_{\pi}(s,a)$
	- combine Sarsa with a policy improvement step.
	- Expected Sarsa:
		- TD target is changed from $r_{t+1}+\gamma q_t(s_{t+1},a_{t+1})$ to $r_{t+1}+\gamma  \mathbb{E}[q_t (s_{t+1},A)]$. Need more computation.
	- $n$-step Sarsa:
		- unify Sarsa and MC learning.
		- Return:$$G_t^{(n)}=R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^n q_{\pi}(S_{t+n},A_{t+n})$$when $n=1$, it is Sarsa; when $n=\infty$, it is MC learning.
		- it needs $(s_t,a_t,r_{t+1},s_{t+1},a_{t+1},\cdots,r_{t+n},s_{t+n},a_{t+n})$
- Q-learning: TD learning of optimal action values
	- Q-learning algorithm:$$q_{t+1}(s_t,a_t)=q_t(s_t,a_t)-\alpha_t(s_t,a_t)\left[q_t(s_t,a_t)-[r_{t+1}+\gamma\max_{a\in A}q_t(s_{t+1},a)]\right],$$$q_{t+1}(s,a)=q_t(s,a), \forall (s,a)\ne (s_t,a_t).$$
	- It solves the Bellman optimality equation.
	- Off-policy vs. On-policy
			- behavior policy: is used to generate experience samples.
			- target policy: is constantly updated toward an optimal policy.
			- On-policy: when the behavior policy is the same as the target policy. (Sarsa, MC learning)
			- Off-policy: when they are different.(Q-learning)
	- can be implemented in an either off-policy or on-policy fashion.
- summary 
	![[Pasted image 20260416195705.png]]![[Pasted image 20260416195759.png]]

## Chapter8. Value Function Approximation
- from tables to functions
	- the tables are difficult to handle large or continuous state or action spaces. 2 aspects: storage and generalization ability.
	- Simplest: straight line$$\hat{v}(s,w)=as+b=\Phi^T(s)w$$where $w$ is the parameter vector; $\phi(s)$ is the feature vector of $s$; $\hat{v}(s,w)$ is linear in $w$.
	- using a second-order curve$$\hat{v}(s,w)=as^2+bs+c=\Phi^T(s)w$$
	- Idea: using parameterized functions:$$\hat{v}(s,w) \approx v_{\pi}(s)$$
	- Advantage: 
		- Storage: the dimension of $w$ may be much less than $|S|$.
		- Generalization: $w$ will be updated so that the values of other unvisited states can also be updated.
- Algorithm for state value estimation
	- objective function: find an optimal $w$.
		- define the objective function$$J(w)=\mathbb{E}[(v_{\pi}(S)-\hat{v}(S,w))^2]$$which is called True value error.
		- several methods:
			- use a uniform distribution:$$J(w)=\frac{1}{|S|}\sum_{s\in S}(v_{\pi}(s)-\hat{v}(s,w))^2.$$but the states may not be equally important.
			- use the stationary distribution:$$J(w)=\sum_{s\in S}d_{\pi}(s)(v_{\pi}(s)-\hat{v}(s,w))^2.$$it describes the long-run behavior of a Markov process.
			- Also called steady-state or limiting distribution.
			- The converged values can be predicted because they are the entries of $d_\pi$: $d_{\pi}^T=d_{\pi}^TP_{\pi}$
	- Optimization algorithms:
		- use stochastic gradient-descent algorithm, but we need $v_{\pi}(s_t)$.
		- Monte Carlo learning with function approximation: let $g_t$ replace it.
		- TD learning with function approximation: let TD target $\left(r_{t+1}+\gamma \hat{v}(s_{t+1},w_t)\right)$ replace it.
	- Selection of function approximators:
		- use a linear function.
		- use a **neural network** as a nonlinear function approximator.
	- Bellman error
	- Projected Bellman error: add M
- Sarsa with function approximation
	- replace $\hat{v}(s_t)$ with $\hat{q}(s_t,a_t)$
	- in value updating we update the parameter
- Q-learning with function approximation
	- replace the TD target
- **Deep Q-learning**: or deep Q-network DQN
	- Objective function/ loss function:$$J(w)=\mathbb{E} \left[\left(R+\gamma \max_{a\in A(S')}\hat{q}(S',a,w)-\hat{q}(S,A,w)\right)^2\right],$$this is actually the Bellman optimality error.
	- Techniques:
		- 2 Networks:
			- main network representing $\hat{q}(s,a,w)$
			- target network $\hat{q}(s,a,w_T)$
			- let $w_T$ fixed
		- Experience replay:
			- replay buffer $B\doteq (s,a,r,s')$
			- it should follow a uniform distribution.
			- break the correlation
			- more sample efficient

## Chapter9. Policy Gradient Methods
- Basic idea
	- represent policies from tables to functions:$$\pi(a|s,\theta)$$
	- scalar metrics: maximize it to get an optimal policy
	- access: calculate
	- update: change the parameter $\theta$
- Metric to define optimal policies
	- average state value:$$\bar{v}_{\pi}=\sum_{s\in S}d(s)v_{\pi}(s)=d^Tv_{\pi}$$
		- select $d$:
			- $d$ is independent of the policy $\pi$:
				- uniform distribution
				- give weight
			- $d$ depends on the policy $\pi$:
				- select $d$ as $d_{\pi}(s)$, stationary distribution
	- average one-step reward:$$\bar{r}_{\pi}\doteq \sum_{s\in S}d_{\pi}(s)r_{\pi}(s)$$
		- equivalent definition:$$\bar{r}_{\pi}=\lim_{n\to \infty}\frac{1}{n}\mathbb{E}\left[\sum_{k=1}^{n}R_{t+k}\right]$$
- Gradient of the metrics
	- summary:$$\begin{aligned}\nabla_{\theta}J(\theta)&=\sum_{s\in S}\eta(s) \sum_{a\in A}\nabla_{\theta}\pi(a|s,\theta)q_{\pi}(s,a)\\&=\mathbb{E}[\nabla_{\theta}ln\pi(A|S,\theta)q_{\pi}(S,A)]\\&\approx \nabla_{\theta}ln\pi(a|s,\theta)q_{\pi}(s,a)\end{aligned}$$
	- softmax functions: normalize the entries in a vector from $(-\infty,+\infty)$ to $(0,1)$
- Gradient-ascent algorithm: REINFORCE
	- maximizing $\pi(a_t|s_t,\theta)$:$$\theta_{t+1}=\theta_t+\alpha\beta_t\nabla_\theta \pu(a_t|s_t,\theta_t)$$where $\beta_t=\frac{q_t(s_t,a_t)}{\pi(a_t|s_t,\theta_t)}$, it can well balance exploration and exploitation.
	- If $q_t(s_t,a_t)$ is approximated by Monte Carlo estimation, it is called REINFORCE.

## Chapter10. Actor-Critic Methods
- Introduction
	- actor: refers to policy update
	- critic: refers to policy evaluation or value estimation
	- It is actually policy gradient
	- If $q_t(s_t,a_t)$ is approximated by TD algorithm, it is AC method
	- The simplest AC: QAC
- Advantage actor-critic: A2C
	- introduce an additional baseline: $b(S)$, $\mathbb{E}(X)$ is invariant to it, but $var(X)$ is not invariant to it.
	- goal: select an optimal baseline to minimize $var(X)$.
	- suboptimal baseline: $b(s)=v_{\pi}(s)$
	- advantage function:$\delta_\pi(S,A)\doteq q_\pi(S,A)-v_\pi(S)$
- Off-policy actor-critic