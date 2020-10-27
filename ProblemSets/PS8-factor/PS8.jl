using MultivariateStats, DataFrames, CSV, HTTP, Random, LinearAlgebra, Statistics, Optim, DataFramesMeta, GLM
include("lgwt.jl")

function ps8()
	#Question 1
	url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS8-factor/nlsy.csv"
	df = CSV.read(HTTP.get(url).body)

	ols = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)
	print(ols)

	#print(coef(ols))


	#Question 2
	asvabMat=convert(Matrix,df[:,r"asvab"])
	correlation=cor(asvabMat)
	#for all variables:
	#cor(convert(Matrix,df))

	println(correlation)



	#Question 3
	ols2 = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr+asvabAR+asvabCS+asvabMK+asvabNO+asvabPC+asvabWK), df)
	print(ols2)



	#I don't know what are these asvab variables but I assume they are some sort of proxy for log wage. They are correlated with each other.
	# if we keep them we have measurement error and if we don't we will have ommited varibale, so any way this models would be biased!

	#Question 4
	#just using transpose instead of reshape
	M = fit(PCA, asvabMat'; maxoutdim=1)

	asvabPCA = MultivariateStats.transform(M, asvabMat')

	print(asvabPCA)

	#Question 5
	M = fit(FactorAnalysis, asvabMat'; maxoutdim=1)
	asvabFA = MultivariateStats.transform(M, asvabMat')

	print(asvabFA)





	#Question 6
	#indexe of M in the formula is correct?
	#getting the random numbers:
	e=rand(Normal(0,1),size(df,1))
	Y=df[:logwage]
	X=df[:,[:black,:hispanic,:female,:schoolt,:gradHS,:grad4yr]]
	asvab=df[:,[:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]]

	function loglike(sigma,m,e,y,x,asv)
		N=size(Y,1)
		J=size(asv,2)
		
		res_abv=zeros(N,J)
		for i=1:J
			res_abv[:,i]=residuals(lm(@formula(x1 ~ black + hispanic + female +x1_1),hcat(x,asvab[i],e,makeunique=true)))
		end
	   
		wage=residuals(lm(@formula(x1 ~ black + hispanic + female + schoolt + gradHS + grad4yr+x1_1), hcat(x,y,e,makeunique=true)))
		
		#how many sigma?
		
		return 
	end
end
ps8()