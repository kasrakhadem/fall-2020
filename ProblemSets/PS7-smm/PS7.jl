using SMM, DataFrames, CSV, HTTP, Random, LinearAlgebra, Statistics, Optim, DataFramesMeta, GLM

function PS7()
	#question 1
	url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
	df = CSV.read(HTTP.get(url).body)
	X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
	y = log.(df.wage);

	function ols_gmm(beta, X, y)
		return (y.-X*beta)'*(y.-X*beta)
	end


	function ols_gmm_with_sigma(beta, X, y)
		g = vcat(y .- X*beta[1:end-1],( (size(y,1)-1)/(size(y,1)-size(X,2)) )*var(y .- X*beta[1:end-1]) .- beta[end]^2)
		return g'*g
	end

	beta_gmm = optimize(b -> ols_gmm(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000))
	println(beta_gmm.minimizer)

	beta_gmm_sigma = optimize(b -> ols_gmm_with_sigma(b, X, y), rand(size(X,2)+1), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000))
	println(beta_gmm_sigma.minimizer)


	#Question 2
	df = dropmissing(df, :occupation)
	df[df.occupation.>7,:occupation] .= 7
	X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
	y = df.occupation;


	#from PS2
	function mlogit_mle(alpha, X, y)
			
			K = size(X,2)
			J = length(unique(y))
			N = length(y)
			bigY = zeros(N,J)
			for j=1:J
				bigY[:,j] = y.==j
			end
			A = [reshape(alpha,K,J-1) zeros(K)]
			
			P = exp.(X*A) ./ sum.(eachrow(exp.(X*A))) 
			
			return -sum( bigY.*log.(P) )
	end

	alpha_hat= optimize(a -> mlogit_mle(a, X, y), rand(6*size(X,2)), LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))

	print(alpha_hat.minimizer)

	function mlogit_gmm(α, X, y)
			K = size(X,2)
			J = length(unique(y))
			N = length(y)
			bigY = zeros(N,J)
			for j=1:J
				bigY[:,j] = y.==j
			end
			bigα = [reshape(α,K,J-1) zeros(K)]
			
			P = exp.(X*bigα) ./ sum.(eachrow(exp.(X*bigα)))
			
			g = bigY[:] .- P[:]

			J = g'*I*g
			return J
	end

	α_hat_gmm = optimize(a -> mlogit_gmm(a, X, y), alpha_hat.minimizer, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=10_000, show_trace=true, show_every=100))
	print(α_hat_gmm.minimizer)

	td = TwiceDifferentiable(b -> mlogit_gmm(b, X, y), alpha_hat.minimizer; autodiff = :forward)
	α_hat_gmm_random=optimize(td, rand(6*size(X,2)), LBFGS(), Optim.Options(g_tol = 1e-5, iterations=10_000, show_trace=true, show_every=100))

	print(α_hat_gmm_random.minimizer)

	#Question 3

	#Question4
	MA = SMM.parallelNormal()
	dc = SMM.history(MA.chains[1])
	dc = dc[dc[:accepted].==true, :]
	println(describe(dc))

	#Question 5
end

PS7()
