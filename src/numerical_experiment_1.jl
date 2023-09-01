#-----------------------Example 1: local reduction---------------------------
using JuMP
using Ipopt
include("local_reduction_pr.jl")

var_size = 3 # dimension of the problem
xL = [-1.0, 0.0, -2.0, -Inf] #lower bound of x
xU = [1.0, 1.0, 0.0, Inf]
yL = [[-1.0, -2.0, -2.0]]
yU = [[1.0, 2.0, 1.0]]
y_k = [zeros(var_size)]

SIP_obj(x) = x[var_size + 1] #Objective function
SIP_cons_1(x,y) = (x[1:var_size] - y)' * (x[1:var_size] - y) - x[var_size + 1]
SIP_cons = [SIP_cons_1]

y_k = [0.5*ones(var_size)]
@time local_reduction_pr(SIP_obj, SIP_cons, xL, xU, yL, yU, y_k; max_iter=50)

#-----------------------------Example 1: EAGO.jl---------------------------------------
using JuMP
using EAGO

var_size = 3 # dimension of the problem
xL = [-1.0, 0.0, -2.0, 0.0] #lower bound of x
xU = [1.0, 1.0, 0.0, 22.0]
yL = [-1.0, -2.0, -2.0]
yU = [1.0, 2.0, 1.0]
SIP_obj(x) = x[var_size + 1] #Objective function
SIP_cons_1(x, y) = ([x[1],x[2],x[3]] - [y[1],y[2],y[3]])' * ([x[1],x[2],x[3]] - [y[1],y[2],y[3]]) - x[var_size + 1]

sip_result = @time sip_solve(SIPRes(), xL, xU, yL, yU, SIP_obj, Any[SIP_cons_1], abs_tolerance = 1e-5)
println("The global minimum of the semi-infinite program is between: $(sip_result.lower_bound) and $(sip_result.upper_bound).")
println("The global minimum is attained at: x = $(sip_result.xsol).")
println("Is the problem feasible? $(sip_result.feasibility).")

#------------------------Example 1: Constraint discretization----------------------------
using JuMP
using Ipopt
include("constraint_discretization.jl")

N = 41
var_size = 3

xL = [-1.0, 0.0, -2.0, -Inf] #lower bound of x
xU = [1.0, 1.0, 0.0, Inf]
yL = [[-1.0, -2.0, -2.0]]
yU = [[1.0, 2.0, 1.0]]
SIP_obj(x) = x[var_size + 1] #Objective function
SIP_cons_1(x,y) = (x[1:var_size] - y)' * (x[1:var_size] - y) - x[var_size + 1]
SIP_cons = [SIP_cons_1]

@time constraint_discretization(SIP_obj, SIP_cons, xL, xU, yL, yU, N)

##########################################################################################
#----------------------------Example 2: local reduction-----------------------------------
using JuMP
using LinearAlgebra
using Ipopt
include("local_reduction_pr.jl")

var_size = 2 #problem dimension

xL = Float64[-10, -20, -Inf]
xU = Float64[5, 25, Inf]
yL = [Float64[0, -3, 2, -25]]
yU = [Float64[3, 4, 6, 5]]
y_k = [Float64[3, 4, 6, 5]]
SIP_obj(x) = x[var_size+1]
SIP_cons_1(x,y) = x[1:var_size]'*diagm(y[1:var_size])*x[1:var_size] + y[var_size+1:2*var_size]'*x[1:var_size] - x[var_size+1]
SIP_cons = [SIP_cons_1]

@time local_reduction_pr(SIP_obj, SIP_cons, xL, xU, yL, yU, y_k; solver = Ipopt, max_iter=50, process_storage = false)

#---------------------------------Example 2: EAGO.jl------------------------------------------
using JuMP
using EAGO
using LinearAlgebra

var_size = 2

xL = Float64[-10, -20, -2560]
xU = Float64[5, 25, 2205]
yL = Float64[0, -3, 2, -25]
yU = Float64[3, 4, 6, 5]
SIP_obj(x) = x[var_size+1]
SIP_cons_1(x,y) = [x[1], x[2]]'*diagm([y[1], y[2]])*[x[1], x[2]] + [y[3], y[4]]'*[x[1], x[2]] - x[var_size+1]

sip_result = @time sip_solve(SIPRes(), xL, xU, yL, yU, SIP_obj, Any[SIP_cons_1], abs_tolerance = 1e-5)
println("The global minimum of the semi-infinite program is between: $(sip_result.lower_bound) and $(sip_result.upper_bound).")
println("The global minimum is attained at: x = $(sip_result.xsol).")
println("Is the problem feasible? $(sip_result.feasibility).")

#---------------------------------Example 2: Discretization------------------------------------
using JuMP
using Ipopt
using LinearAlgebra
include("constraint_discretization.jl")

N = Int64(50)
var_size = 2

xL = Float64[-10, -20, -Inf]
xU = Float64[5, 25, Inf]
yL = [Float64[0, -3, 2, -25]]
yU = [Float64[3, 4, 6, 5]]
SIP_obj(x) = x[var_size+1]
SIP_cons_1(x,y) = x[1:var_size]'*diagm(y[1:var_size])*x[1:var_size] + y[var_size+1:2*var_size]'*x[1:var_size] - x[var_size+1]
SIP_cons = [SIP_cons_1]

@time constraint_discretization(SIP_obj, SIP_cons, xL, xU, yL, yU, N)

################################################################################################
#-------------------------Example 3 - Trid Function: Local reduction----------------------------
using JuMP
using Ipopt
include("local_reduction_pr.jl")

var_size = 3

xL = [-9*ones(Float64, var_size); -Inf]
xU = [9*ones(Float64, var_size); Inf]
yL = [-ones(Float64, var_size)]
yU = [ones(Float64, var_size)]
y_k = [zeros(Float64, var_size)]

SIP_obj(x) = x[var_size+1]
SIP_cons_1(x,y) = sum(y[i]*(x[i]-1)^2 for i in 1:var_size) - sum(y[i]*x[i]*x[i-1] for i in 2:var_size) - x[var_size+1]
SIP_cons = [SIP_cons_1]

@time local_reduction_pr(SIP_obj, SIP_cons, xL, xU, yL, yU, y_k; solver = Ipopt, max_iter=50, process_storage = false)

#----------------------------------------Example 3: EAGO.jl-------------------------------------
using JuMP
using EAGO

var_size = 3

xL = [-9*ones(Float64, var_size); -426]
xU = [9*ones(Float64, var_size); 426]
yL = -ones(Float64, var_size)
yU = ones(Float64, var_size)

SIP_obj(x) = x[var_size+1]
SIP_cons_1(x,y) = sum(y[i]*(x[i]-1)^2 for i in 1:var_size) - sum(y[i]*x[i]*x[i-1] for i in 2:var_size) - x[var_size+1]

sip_result = @time sip_solve(SIPRes(), xL, xU, yL, yU, SIP_obj, Any[SIP_cons_1], abs_tolerance = 1e-5)
println("The global minimum of the semi-infinite program is between: $(sip_result.lower_bound) and $(sip_result.upper_bound).")
println("The global minimum is attained at: x = $(sip_result.xsol).")
println("Is the problem feasible? $(sip_result.feasibility).")

#-------------------------------------Example 3: discretization----------------------------------
using JuMP
using Ipopt
using LinearAlgebra
include("constraint_discretization.jl")

N = Int64(20)
var_size = 3
xL = [-9*ones(Float64, var_size); -Inf]
xU = [9*ones(Float64, var_size); Inf]
yL = [-ones(Float64, var_size)]
yU = [ones(Float64, var_size)]

SIP_obj(x) = x[var_size+1]
SIP_cons_1(x,y) = sum(y[i]*(x[i]-1)^2 for i in 1:var_size) - sum(y[i]*x[i]*x[i-1] for i in 2:var_size) - x[var_size+1]
SIP_cons = [SIP_cons_1]

@time constraint_discretization(SIP_obj, SIP_cons, xL, xU, yL, yU, N)

###############################################################################################
#--------------------Example 4 - Styblinski-Tang Function: Local reduction ---------------------
using JuMP
using Ipopt
using LinearAlgebra
include("local_reduction_pr.jl")

var_size = 3

xL = [-10*ones(Float64, var_size); -Inf]
xU = [10*ones(Float64, var_size); Inf]
yL = [-ones(Float64, var_size)]
yU = [ones(Float64, var_size)]
y_k = [ones(Float64, var_size)]

SIP_obj(x) = x[var_size+1]
SIP_cons_1(x,y) = 0.5*((x[1:var_size].^2)'*diagm(y)*x[1:var_size].^2 - 16*x[1:var_size]'*diagm(y)*x[1:var_size] + 5*x[1:var_size]'*ones(var_size)) - x[var_size+1]
SIP_cons = [SIP_cons_1]

@time local_reduction_pr(SIP_obj, SIP_cons, xL, xU, yL, yU, y_k; solver = Ipopt, max_iter=50, abs_tolerance=1e-5)

#-------------------------------------Example 4: EAGO.jl---------------------------------------
using JuMP
using EAGO
using LinearAlgebra

var_size = 3

xL = [-10*ones(Float64, var_size); -12675]
xU = [10*ones(Float64, var_size); 12675]
yL = -ones(Float64, var_size)
yU = ones(Float64, var_size)

SIP_obj(x) = x[var_size+1]
SIP_cons_1(x,y) = 0.5*(([x[1],x[2],x[3]].^2)'*diagm([y[1],y[2],y[3]])*[x[1],x[2],x[3]].^2 - 16*[x[1],x[2],x[3]]'*diagm([y[1],y[2],y[3]])*[x[1],x[2],x[3]] + 5*[x[1],x[2],x[3]]'*ones(var_size)) - x[var_size+1]

sip_result = @time sip_solve(SIPRes(), xL, xU, yL, yU, SIP_obj, Any[SIP_cons_1], abs_tolerance = 1e-5)
println("The global minimum of the semi-infinite program is between: $(sip_result.lower_bound) and $(sip_result.upper_bound).")
println("The global minimum is attained at: x = $(sip_result.xsol).")
println("Is the problem feasible? $(sip_result.feasibility).")

#-----------------------------------Example 4: Discretization-----------------------------------
using JuMP
using Ipopt
using LinearAlgebra
include("constraint_discretization.jl")

N = Int64(21)
var_size = 5
xL = [-10*ones(Float64, var_size); -Inf]
xU = [10*ones(Float64, var_size); Inf]
yL = [-ones(Float64, var_size)]
yU = [ones(Float64, var_size)]

SIP_obj(x) = x[var_size+1]
SIP_cons_1(x,y) = 0.5*((x[1:var_size].^2)'*diagm(y)*x[1:var_size].^2 - 16*x[1:var_size]'*diagm(y)*x[1:var_size] + 5*x[1:var_size]'*ones(var_size)) - x[var_size+1]
SIP_cons = [SIP_cons_1]

@time constraint_discretization(SIP_obj, SIP_cons, xL, xU, yL, yU, N; solver=Ipopt)


###########################################################################################################
#---------------------------------------Example 5: local reduction-----------------------------------------
using JuMP
using Ipopt
include("local_reduction_pr.jl")

var_size = 50

xL = zeros(Float64, var_size)
xU = ones(Float64, var_size)
yL = [zeros(Float64, var_size), zeros(Float64, var_size)]
yU = [ones(Float64, var_size), ones(Float64, var_size)]
y_k = [0.5*ones(Float64, var_size), 0.8*ones(Float64, var_size)]

SIP_obj(x) = sum(x[i] for i in 1:var_size)
SIP_cons_1(x,y) = -sum(x[i]*(y[i]-2)^2 for i in 1:var_size) + 50
SIP_cons_2(x,y) = -sum(x[i]^0.5*(y[i]-2)^2 for i in 1:var_size) + 50
SIP_cons = [SIP_cons_1, SIP_cons_2]

@time local_reduction_pr(SIP_obj, SIP_cons, xL, xU, yL, yU, y_k; solver = Ipopt, max_iter=50, abs_tolerance=1e-5)

#----------------------------------------Example 5: EAGO.jl--------------------------------------------------
using JuMP
using EAGO

var_size = 50

xL = 1e-5*ones(Float64, var_size)
xU = ones(Float64, var_size)
yL = 1e-5*ones(Float64, var_size)
yU = ones(Float64, var_size)

SIP_obj(x) = sum(x[i] for i in 1:var_size)
SIP_cons_1(x,y) = -sum(x[i]*(y[i] - 2)^2 for i in 1:var_size) + 50
SIP_cons_2(x,y) = -sum(x[i]^0.5*(y[i] - 2)^2 for i in 1:var_size) + 50

sip_result = @time sip_solve(SIPRes(), xL, xU, yL, yU, SIP_obj, Any[SIP_cons_1, SIP_cons_2], abs_tolerance = 1e-5)
println("The global minimum of the semi-infinite program is between: $(sip_result.lower_bound) and $(sip_result.upper_bound).")
println("The global minimum is attained at: x = $(sip_result.xsol).")
println("Is the problem feasible? $(sip_result.feasibility).")

#-----------------------------------------Example 5: discretization---------------------------------------------
using JuMP
using Ipopt
include("constraint_discretization.jl")

N = Int64(11)
var_size = 50

xL = zeros(Float64, var_size)
xU = ones(Float64, var_size)
yL = [zeros(Float64, var_size), zeros(Float64, var_size)]
yU = [ones(Float64, var_size), ones(Float64, var_size)]

SIP_obj(x) = sum(x[i] for i in 1:var_size)
SIP_cons_1(x,y) = -sum(x[i]*(y[i]-2)^2 for i in 1:var_size) + 50
SIP_cons_2(x,y) = -sum(x[i]^0.5*(y[i]-2)^2 for i in 1:var_size) + 50
SIP_cons = [SIP_cons_1, SIP_cons_2]

@time constraint_discretization(SIP_obj, SIP_cons, xL, xU, yL, yU, N; solver=Ipopt)


#######################################################################################################
#------------------------------------Example 6: local reduction----------------------------------------
using JuMP
using Ipopt
include("local_reduction_pr.jl")

var_size = 4

xL = [-pi*ones(Float64, var_size); -Inf]
xU = [pi*ones(Float64, var_size); Inf]
yL = [-1.0*ones(Float64, var_size)]
yU = [1.0*ones(Float64, var_size)]
y_k = [-0.5*ones(Float64, var_size)]

SIP_obj(x) = x[var_size+1]
SIP_cons_1(x,y) = sum((y[i]+0.1*i)*sin(x[i]) for i in 1:var_size) - y'*cos.(x[1:var_size]) - x[var_size+1]
SIP_cons = [SIP_cons_1]

@time local_reduction_pr(SIP_obj, SIP_cons, xL, xU, yL, yU, y_k; solver = Ipopt, max_iter=50, abs_tolerance=1e-5)

#---------------------------------------Example 6: EAGO.jl------------------------------------------------
using JuMP
using EAGO

var_size = 4

xL = [-pi*ones(Float64, var_size); -6.41]
xU = [pi*ones(Float64, var_size); 5.02]
yL = -1.0*ones(Float64, var_size)
yU = 1.0*ones(Float64, var_size)

SIP_obj(x) = x[var_size+1]
SIP_cons_1(x,y) = sum((y[i]+0.1*i)*sin(x[i]) - y[i]*cos(x[i]) for i in 1:var_size) - x[var_size+1]

sip_result = @time sip_solve(SIPRes(), xL, xU, yL, yU, SIP_obj, Any[SIP_cons_1], abs_tolerance = 1e-5)
println("The global minimum of the semi-infinite program is between: $(sip_result.lower_bound) and $(sip_result.upper_bound).")
println("The global minimum is attained at: x = $(sip_result.xsol).")
println("Is the problem feasible? $(sip_result.feasibility).")

#--------------------------------------Example 6: Discretization-------------------------------------------
using JuMP
using Ipopt
include("constraint_discretization.jl")

N = Int64(21)
var_size = 4
xL = [-pi*ones(Float64, var_size); -Inf]
xU = [pi*ones(Float64, var_size); Inf]
yL = [-1.0*ones(Float64, var_size)]
yU = [1.0*ones(Float64, var_size)]

SIP_obj(x) = x[var_size+1]
SIP_cons_1(x,y) = sum((y[i]+0.1*i)*sin(x[i]) - y[i]*cos(x[i]) for i in 1:var_size) - x[var_size+1]
SIP_cons = [SIP_cons_1]

@time constraint_discretization(SIP_obj, SIP_cons, xL, xU, yL, yU, N; solver=Ipopt)

###########################################################################################################
#---------------------------------------Example 7 - hart6: local reduction-----------------------------------------
using JuMP
using Ipopt
include("local_reduction_pr.jl")

var_size = 2

xL = [zeros(Float64, var_size); -Inf]
xU = [ones(Float64, var_size); Inf]
yL = [-1.0*ones(Float64, var_size)]
yU = [1.0*ones(Float64, var_size)]
y_k = [-0.7*ones(Float64, var_size)]

SIP_obj(x) = x[var_size+1]
SIP_cons_1(x,y) = -exp(-(10* ((-0.1312)+y[1]*x[1])^2+0.05* ((-0.1696)+y[2]*x[2])^2))+1.2*exp(-(0.05* ((-0.2329)+y[1]*x[1])^2+10* ((-0.4135)+y[2]*x[2])^2))+3*exp(-(3* ((-0.2348)+y[1]*x[1])^2+3.5* ((-0.1451)+y[2]*x[2])^2)) - x[var_size+1]
SIP_cons = [SIP_cons_1]

@time local_reduction_pr(SIP_obj, SIP_cons, xL, xU, yL, yU, y_k; solver = Ipopt, max_iter=50, abs_tolerance=1e-5)

#---------------------------------------Example 7: EAGO.jl------------------------------------------------
using JuMP
using EAGO

var_size = 2

xL = [zeros(Float64, var_size); -10]
xU = [ones(Float64, var_size); 10]
yL = -1.0*ones(Float64, var_size)
yU = 1.0*ones(Float64, var_size)

SIP_obj(x) = x[var_size+1]
SIP_cons_1(x,y) = -exp(-(10* ((-0.1312)+y[1]*x[1])^2+0.05* ((-0.1696)+y[2]*x[2])^2))+1.2*exp(-(0.05* ((-0.2329)+y[1]*x[1])^2+10* ((-0.4135)+y[2]*x[2])^2))+3*exp(-(3* ((-0.2348)+y[1]*x[1])^2+3.5* ((-0.1451)+y[2]*x[2])^2)) - x[var_size+1]

sip_result = @time sip_solve(SIPRes(), xL, xU, yL, yU, SIP_obj, Any[SIP_cons_1], abs_tolerance = 1e-5)
println("The global minimum of the semi-infinite program is between: $(sip_result.lower_bound) and $(sip_result.upper_bound).")
println("The global minimum is attained at: x = $(sip_result.xsol).")
println("Is the problem feasible? $(sip_result.feasibility).")

#--------------------------------------Example 7: Discretization-------------------------------------------
using JuMP
using Ipopt
include("constraint_discretization.jl")

N = Int64(21)
var_size = 2
xL = [zeros(Float64, var_size); -Inf]
xU = [ones(Float64, var_size); Inf]
yL = [-1.0*ones(Float64, var_size)]
yU = [1.0*ones(Float64, var_size)]


SIP_obj(x) = x[var_size+1]
SIP_cons_1(x,y) = -exp(-(10* ((-0.1312)+y[1]*x[1])^2+0.05* ((-0.1696)+y[2]*x[2])^2))+1.2*exp(-(0.05* ((-0.2329)+y[1]*x[1])^2+10* ((-0.4135)+y[2]*x[2])^2))+3*exp(-(3* ((-0.2348)+y[1]*x[1])^2+3.5* ((-0.1451)+y[2]*x[2])^2)) - x[var_size+1]
SIP_cons = [SIP_cons_1]

@time constraint_discretization(SIP_obj, SIP_cons, xL, xU, yL, yU, N; solver=Ipopt)

###########################################################################################################
#---------------------------------------Example 8: local reduction-----------------------------------------
using JuMP
using Ipopt
include("local_reduction_pr.jl")

var_size = 2

xL = zeros(Float64, var_size)
xU = [3.0, 4.0]
yL = [[0.8], [-1.0]]
yU = [[1.2], [1.0]]
y_k = [[1.1], [-0.5]]

SIP_obj(x) = -x[1] -x[2]
SIP_cons_1(x,y) = 8* (x[1]*y[1])^3-2* (x[1]*y[1])^4-8* (x[1]*y[1])^2+x[2] - 2.0
SIP_cons_2(x,y) = 32* (x[1]*y[1])^3-4* (x[1]*y[1])^4-88* (x[1]*y[1])^2+96*x[1]+x[2] - 36.0
SIP_cons = [SIP_cons_1, SIP_cons_2]

@time local_reduction_pr(SIP_obj, SIP_cons, xL, xU, yL, yU, y_k; solver = Ipopt, max_iter=50, abs_tolerance=1e-5)

#---------------------------------------Example 8: EAGO.jl------------------------------------------------
using JuMP
using EAGO

var_size = 2

xL = zeros(Float64, var_size)
xU = [3.0, 4.0]
yL = [0.8, -1.0]
yU = [1.2, 1.0]

SIP_obj(x) = -x[1] -x[2]
SIP_cons_1(x,y) = 8* (x[1]*y[1])^3-2* (x[1]*y[1])^4-8* (x[1]*y[1])^2+x[2] - 2.0
SIP_cons_2(x,y) = 32* (x[1]*y[2])^3-4* (x[1]*y[2])^4-88* (x[1]*y[2])^2+96*x[1]+x[2] - 36.0

sip_result = @time sip_solve(SIPRes(), xL, xU, yL, yU, SIP_obj, Any[SIP_cons_1, SIP_cons_2], abs_tolerance = 1e-5)
println("The global minimum of the semi-infinite program is between: $(sip_result.lower_bound) and $(sip_result.upper_bound).")
println("The global minimum is attained at: x = $(sip_result.xsol).")
println("Is the problem feasible? $(sip_result.feasibility).")

#--------------------------------------Example 8: Discretization-------------------------------------------
using JuMP
using Ipopt
include("constraint_discretization.jl")

N = Int64(21)
var_size = 2
xL = zeros(Float64, var_size)
xU = [3.0, 4.0]
yL = [[0.8], [-1.0]]
yU = [[1.2], [1.0]]

SIP_obj(x) = -x[1] -x[2]
SIP_cons_1(x,y) = 8* (x[1]*y[1])^3-2* (x[1]*y[1])^4-8* (x[1]*y[1])^2+x[2] - 2.0
SIP_cons_2(x,y) = 32* (x[1]*y[1])^3-4* (x[1]*y[1])^4-88* (x[1]*y[1])^2+96*x[1]+x[2] - 36.0
SIP_cons = [SIP_cons_1, SIP_cons_2]

@time constraint_discretization(SIP_obj, SIP_cons, xL, xU, yL, yU, N)

###########################################################################################################
#---------------------------------------Example 9: local reduction-----------------------------------------
using JuMP
using Ipopt
include("local_reduction_pr.jl")

var_size = 2

xL = [-10.0*ones(Float64, var_size); -Inf]
xU = [10.0*ones(Float64, var_size); Inf]
yL = [[-1.0, 0.8]]
yU = [[1.0, 1.2]]
y_k = [[1.0, 1.2]]

SIP_obj(x) = x[var_size+1]
SIP_cons_1(x,y) = (-5 + (x[1]*y[1])^2) * (-5 + (x[1]*y[2])^3) + (-5 + (x[2]*y[2])^2) * (-5 + (x[2]*y[1])^3) - x[var_size+1]
SIP_cons = [SIP_cons_1]

@time local_reduction_pr(SIP_obj, SIP_cons, xL, xU, yL, yU, y_k; solver = Ipopt, max_iter=10, abs_tolerance=1e-5)

#---------------------------------------Example 9: EAGO.jl------------------------------------------------
using JuMP
using EAGO

var_size = 2

xL = [-10.0*ones(Float64, var_size); -1e4]
xU = [10.0*ones(Float64, var_size); 1e4]
yL = [-1.0, 0.8]
yU = [1.0, 1.2]

SIP_obj(x) = x[var_size+1]
SIP_cons_1(x,y) = (-5 + (x[1]*y[1])^2) * (-5 + (x[1]*y[2])^3) + (-5 + (x[2]*y[2])^2) * (-5 + (x[2]*y[1])^3) - x[var_size+1]

sip_result = @time sip_solve(SIPRes(), xL, xU, yL, yU, SIP_obj, Any[SIP_cons_1], abs_tolerance = 1e-5)
println("The global minimum of the semi-infinite program is between: $(sip_result.lower_bound) and $(sip_result.upper_bound).")
println("The global minimum is attained at: x = $(sip_result.xsol).")
println("Is the problem feasible? $(sip_result.feasibility).")

#---------------------------------------Example 9: discretization---------------------------------------
using JuMP
using Ipopt
include("constraint_discretization.jl")

N = Int64(21)
var_size = 2
xL = [-10.0*ones(Float64, var_size); -Inf]
xU = [10.0*ones(Float64, var_size); Inf]
yL = [[-1.0, 0.8]]
yU = [[1.0, 1.2]]


SIP_obj(x) = x[var_size+1]
SIP_cons_1(x,y) = (-5 + (x[1]*y[1])^2) * (-5 + (x[1]*y[2])^3) + (-5 + (x[2]*y[2])^2) * (-5 + (x[2]*y[1])^3) - x[var_size+1]
SIP_cons = [SIP_cons_1]

@time constraint_discretization(SIP_obj, SIP_cons, xL, xU, yL, yU, N; solver=Ipopt)

###########################################################################################################
#---------------------------------------Example 10: local reduction----------------------------------------
using JuMP
using Ipopt
include("local_reduction_pr.jl")

xL = Float64[0, 0, -0.55, -0.55]
xU = Float64[1200, 1200, 0.55, 0.55]
yL = [[0.8] for i in 1:4]
yU = [[1.2] for i in 1:4]
y_k = [[0.8] for i in 1:4]

SIP_obj(x) = 3*x[1] + 1e-6*x[1]^3 + 2*x[2] + (2e-6)/3*x[2]^3
SIP_cons_1(x,y) = y[1]*(-x[3] + x[4]) - 0.55
SIP_cons_2(x,y) = 1e3*sin(-x[3]-0.25) + 1e3*y[1]*sin(-x[4]-0.25) + 894.8 - x[1]
SIP_cons_3(x,y) = 1e3*sin(x[3]-0.25) + 1e3*y[1]*sin(x[3]-x[4]-0.25) + 894.8 - x[2]
SIP_cons_4(x,y) = 1e3*sin(x[4]-0.25) + 1e3*y[1]*sin(x[4]-x[3]-0.25) + 1294.8
SIP_cons = [SIP_cons_1, SIP_cons_2, SIP_cons_3, SIP_cons_4]

@time local_reduction_pr(SIP_obj, SIP_cons, xL, xU, yL, yU, y_k; solver = Ipopt, max_iter=50, abs_tolerance=1e-5, process_storage = false)

#----------------------------------------Example 10: EAGO.jl------------------------------------------------
using JuMP
using EAGO

xL = Float64[0, 0, -0.55, -0.55]
xU = Float64[1200, 1200, 0.55, 0.55]
yL = [0.8]
yU = [1.2]

SIP_obj(x) = 3*x[1] + 1e-6*x[1]^3 + 2*x[2] + (2e-6)/3*x[2]^3
SIP_cons_1(x,y) = y[1]*(-x[3] + x[4]) - 0.55
SIP_cons_2(x,y) = 1e3*sin(-x[3]-0.25) + 1e3*y[1]*sin(-x[4]-0.25) + 894.8 - x[1]
SIP_cons_3(x,y) = 1e3*sin(x[3]-0.25) + 1e3*y[1]*sin(x[3]-x[4]-0.25) + 894.8 - x[2]
SIP_cons_4(x,y) = 1e3*sin(x[4]-0.25) + 1e3*y[1]*sin(x[4]-x[3]-0.25) + 1294.8

sip_result = @time sip_solve(SIPRes(), xL, xU, yL, yU, SIP_obj, Any[SIP_cons_1, SIP_cons_2, SIP_cons_3, SIP_cons_4], abs_tolerance = 1e-5)
println("The global minimum of the semi-infinite program is between: $(sip_result.lower_bound) and $(sip_result.upper_bound).")
println("The global minimum is attained at: x = $(sip_result.xsol).")
println("Is the problem feasible? $(sip_result.feasibility).")

#--------------------------------------Example 10: Discretization-------------------------------------------
using JuMP
using Ipopt
include("constraint_discretization.jl")

N = Int64(5)

xL = Float64[0, 0, -0.55, -0.55]
xU = Float64[1200, 1200, 0.55, 0.55]
yL = [[0.8] for i in 1:4]
yU = [[1.2] for i in 1:4]

SIP_obj(x) = 3*x[1] + 1e-6*x[1]^3 + 2*x[2] + (2e-6)/3*x[2]^3
SIP_cons_1(x,y) = y[1]*(-x[3] + x[4]) - 0.55
SIP_cons_2(x,y) = 1e3*sin(-x[3]-0.25) + 1e3*y[1]*sin(-x[4]-0.25) + 894.8 - x[1]
SIP_cons_3(x,y) = 1e3*sin(x[3]-0.25) + 1e3*y[1]*sin(x[3]-x[4]-0.25) + 894.8 - x[2]
SIP_cons_4(x,y) = 1e3*sin(x[4]-0.25) + 1e3*y[1]*sin(x[4]-x[3]-0.25) + 1294.8
SIP_cons = [SIP_cons_1, SIP_cons_2, SIP_cons_3, SIP_cons_4]

@time constraint_discretization(SIP_obj, SIP_cons, xL, xU, yL, yU, N; solver=Ipopt)
