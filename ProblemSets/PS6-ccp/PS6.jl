using DataFrames, CSV, HTTP, Random, LinearAlgebra, Statistics, Optim, DataFramesMeta, GLM

include("create_grids.jl")

function PS6()
#Question 1

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS5-ddc/busdata.csv"
df = CSV.read(HTTP.get(url).body)



df = @transform(df, bus_id = 1:size(df,1))

#---------------------------------------------------
# reshape from wide to long (must do this twice be-
# cause DataFrames.stack() requires doing it one 
# variable at a time)
#---------------------------------------------------
# first reshape the decision variable
dfy = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded)
dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
rename!(dfy_long, :value => :Y)
dfy_long = @transform(dfy_long, time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfy_long, Not(:variable))

# next reshape the odometer variable
dfx = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
rename!(dfx_long, :value => :Odometer)
dfx_long = @transform(dfx_long, time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfx_long, Not(:variable))

# join reshaped df's back together
df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
sort!(df_long,[:bus_id,:time])

#question 2

θ=glm(@formula(Y ~ Odometer * Odometer^2 * RouteUsage * RouteUsage^2 * Branded * time * time^2), df_long, Binomial(), LogitLink())

#Question 3

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS5-ddc/busdata.csv"
df2 = CSV.read(HTTP.get(url).body)
Y = Matrix(df2[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])
X = Matrix(df2[:,[:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20]])
Z = Vector(df2[:,:RouteUsage])
B = Vector(df2[:,:Branded])
N = size(Y,1)
T = size(Y,2)
Xstate = Matrix(df2[:,[:Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20]])
Zstate = Vector(df2[:,:Zst])
zval,zbin,xval,xbin,xtran = create_grids()


log_df=DataFrame()
log_df.Odometer=kron(ones(zbin),xval) #20301
log_df.RouteUsage =kron(ones(xbin),zval) #20301
log_df.Branded=zeros((size(log_df,1)))
log_df.time=zeros(size(log_df,1))




function values(din,Zstate,Xstate,xtran,zbin,xbin,xval)
    FV=zeros(size(xtran,1),2,T+1)
    for t=2:T
        for b=0:1
            din.time[t]=t
            din.Branded[t]=b
            p0 = predict(θ, din)
            
            FV[:, b+1, t] = - .9 .*log.(p0)
        end
    end
    FVT1 = zeros(size(df,1), T)
    for i=1:size(df,1)
        row0 = (Zstate[i]-1)*xbin+1
        for t=1:T
            row1  = Xstate[i,t] + (Zstate[i]-1)*xbin
            FVT1[i,t] = (xtran[row1,:].-xtran[row0,:])'*FV[row0:row0+xbin-1,df.Branded[i]+1,t+1]
        end
    end
            
    return FVT1'[:]
end
    

fvt1=values(log_df,Zstate,Xstate,xtran,zbin,xbin,xval)
df_long = @transform(df_long,fv = fvt1)

theta_hat_ccp_glm = glm(@formula(Y ~ Odometer + Branded),df_long, Binomial(), LogitLink(),offset=df_long.fv)

end

@time PS6()

