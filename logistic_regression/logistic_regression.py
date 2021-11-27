"""
Logistic Regression Hypothesis:
hΘ = g(z) where z = weighted sum and g = sigmoid function
(also known as logistic function) = 1 / (1 + e^-z)

Logistic Regresion Cost Func:
J(Θ) = 1 / m Σ_i=1 to m Cost(hΘ(xi, yi)) where
Cost(hΘ(xi, yi)) = 
	-log(hΘ_xi) if y = 1
	-log(1 - hΘ_xi) if y = 0
or
J(Θ) = -1 / m [Σ_i=1 to m yi * log(hΘ_xi) + (1 - yi) * log(1 - hΘ_xi)]

The partial derivative term in GD stays the same:
ddΘ_j = 1/m Σ_i=1 to m ((h_Θ(xi) - y_i))xi_j

Learnings:
- The only real difference between linear regression and logistic
regression is that the definition of the hypothesis changed (the
weighted sum is now passed through sigmoid). Other than that,
everything else stays the same including GD.

Things to do:
- regularize 
"""