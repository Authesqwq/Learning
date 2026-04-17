supervised learning vs. unsupervised learning
recommender systems, reinforcement learning

## Part 1. ML basic knowledge
- Supervised Learning
	inputs $\to$ outputs
	Regression
	Classification
- Unsupervised Learning
	find something unlabeled
	Clustering
	only inputs
	Anomaly detection
	Dimensionality reduction
- Linear regression
	- Terminology:
		- trainingset:
			- $x$: input variable-features
			- $y$: output variable-targets
			- $m$: number of training examples
		- $x\to f\to \hat{y}$
		- $f_{w,b}(x)=wx+b$
		- parameters,coefficients,weights
	- Cost function:
		- $\hat{y}^{(i)}=f_{w,b}(x^(i))=wx^(i)+b$
		- squared error cost function: $J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}(\hat{y}^{(i)}-y^{(i)})^2$
		- goal: $minimize_{w,b}J(w,b)$
	- Gradient descent:
		- $w=w-\alpha \frac{\partial}{\partial w}J(w,b)$, = is assignment
		- $\alpha$ is learning rate
		- simultaneous update
		- it cannot update the local minimum
	- Multiple features(variables)
		- $f(x)=\overrightarrow{w} X+b$
	- Vectorization:
		- code counts from 0
		- np.array;np.dot
	- Feature scaling
		- mean normalization:$$x_i=\frac{x_i-\mu_i}{max_i-min_i}$$
		- Z-score normalization:$$x_i=\frac{x_i-\mu_i}{\sigma_i}$$
	- check convergence:
		- introduce $\epsilon$
		- choose appropriate $\alpha$
	- Feature engineering:
		- use intuition to design new variables
	- Polynomial regression
		- use $x^2,x^3,\dots,\sqrt{x}$
	- Logistic regression
		- outputs between 0 and 1
		- sigmoid function(logistic function):$$g(z)=\frac{1}{1+e^{-z}}$$
		- combine them:$$f(X)=g(w\cdot  X+b)$$
		- Desicion boundary: $z=0$
		- Cost function:
			- loss function:$$L(f(x^{(i)}),y^{(i)})=\begin{cases} -log(f(x^{i})), \; if \; y^{(i)}=1\\-log(1-f(x^{(i)})), \; if\; y^{(i)}=0 \end{cases}$$
			- cost function:$$J(w,b)=\frac{1}{m}\sum_{i=1}^{m}L(f(x^{(i)}),y^{(i)})$$
			- simplified loss function:$$L(f(x),y)=-ylog(f(x))-(1-y)log(1-f(x))$$
			- simplified cost function
	- The problem of overfitting
		- problems：
			- underfit and high bias
			- high variance
		- addressing:
			- collect more training examples
			- select features to include/exclude
			- regularization
				- penalizing the model:$$J(w,b)=\frac{1}{2m}\sum(f(x)-y)^2+\frac{\lambda}{2m}\sum{w_j^2}$$ where $\lambda$ is the regularization term.
				- Regularized linear regression:
					- $+\frac{\lambda}{m}w_j$
					- don't have to regularize $b$
				- Regularized logistic regression:
					- same as LR

## Part 2. Advanced learning algorithms
- Neurons and brain
	- Neural networks: algorithms that try to mimic the brain
		- speech$\to$images$\to$text(NLP)
		- $imputs\to \bigcirc \to outputs$
		- layer: can have multiple neurons
		- 